#!/usr/bin/env python3
"""A small, dependency-free Gaussian-process emulator for the rcBK+CGC HERA fit.

Two pieces:
  * GP        -- anisotropic squared-exponential Gaussian-process regression with
                 hyperparameters (per-dim length scales, signal & noise variance)
                 fit by maximizing the log marginal likelihood.  Pure NumPy/SciPy
                 (sklearn is ABI-broken in this environment).
  * ShapeEmulator -- emulates the 271-dim reduced-cross-section SHAPE vector
                 g(theta) via PCA + one GP per principal-component score, so the
                 chi^2 (with sigma0/2 profiled analytically) is exact given the
                 emulated shape.  Emulating the smooth physics rather than the
                 sharp chi^2 valley makes the surrogate far more accurate.

Inputs theta are handled in a user-supplied TRANSFORMED space (e.g. log for
positive parameters); the GP standardizes internally.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize
from scipy.stats import qmc


# =============================================================================
#  Latin-hypercube design in a box of transformed parameters
# =============================================================================
def lhs_design(n, bounds, seed=0):
    """n points in the box `bounds` = [(lo,hi), ...] (already in transformed space)."""
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    u = qmc.LatinHypercube(d=len(bounds), seed=seed).random(n)
    return lo + u * (hi - lo)


# =============================================================================
#  Anisotropic squared-exponential Gaussian process
# =============================================================================
class GP:
    def __init__(self, n_restarts=6, jitter=1e-8, seed=0):
        self.n_restarts = n_restarts
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)

    # ---- kernel: sf2 * exp(-0.5 * sum_d (x_d - x'_d)^2 / l_d^2) --------------
    @staticmethod
    def _sqdist(A, B):
        return (np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T)

    def _nll(self, theta, X, y):
        d = X.shape[1]
        l = np.exp(theta[:d]); sf2 = np.exp(2 * theta[d]); sn2 = np.exp(2 * theta[d + 1])
        Xs = X / l
        K = sf2 * np.exp(-0.5 * np.clip(self._sqdist(Xs, Xs), 0, None))
        K[np.diag_indices_from(K)] += sn2 + self.jitter
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e25
        alpha = cho_solve((L, True), y)
        return float(0.5 * y @ alpha + np.sum(np.log(np.diag(L)))
                     + 0.5 * len(X) * np.log(2 * np.pi))

    def fit(self, X, y):
        # standardize inputs and output
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-12
        self.ym, self.ysd = y.mean(), y.std() + 1e-12
        Xn = (X - self.xm) / self.xs
        yn = (y - self.ym) / self.ysd
        d = Xn.shape[1]
        # bounds on log-hyperparams: length scales, log sf, log sn
        bnds = [(-3.0, 3.0)] * d + [(-3.0, 2.0), (-6.0, 0.5)]
        best = None
        for i in range(self.n_restarts):
            th0 = np.array([self.rng.uniform(lo, hi) for lo, hi in bnds])
            if i == 0:
                th0 = np.array([0.0] * d + [0.0, -2.0])          # sensible default start
            r = minimize(self._nll, th0, args=(Xn, yn), method="L-BFGS-B", bounds=bnds)
            if best is None or r.fun < best.fun:
                best = r
        th = best.x
        self.l = np.exp(th[:d]); self.sf2 = np.exp(2 * th[d]); self.sn2 = np.exp(2 * th[d + 1])
        Xs = Xn / self.l
        K = self.sf2 * np.exp(-0.5 * np.clip(self._sqdist(Xs, Xs), 0, None))
        K[np.diag_indices_from(K)] += self.sn2 + self.jitter
        self.L = cholesky(K, lower=True)
        self.alpha = cho_solve((self.L, True), yn)
        self.Xn = Xn
        self.nll_ = best.fun
        return self

    def predict(self, Xstar, return_std=False):
        Xn = (np.atleast_2d(Xstar) - self.xm) / self.xs
        Ks = self.sf2 * np.exp(-0.5 * np.clip(self._sqdist(Xn / self.l, self.Xn / self.l), 0, None))
        mean = (Ks @ self.alpha) * self.ysd + self.ym
        if not return_std:
            return mean
        v = cho_solve((self.L, True), Ks.T)
        var = self.sf2 + self.sn2 - np.einsum("ij,ji->i", Ks, v)
        return mean, np.sqrt(np.clip(var, 0, None)) * self.ysd


# =============================================================================
#  Shape emulator: g(theta) in R^Npts via PCA + per-PC GPs;  exact chi^2
# =============================================================================
class ShapeEmulator:
    def __init__(self, var_keep=0.9999, max_pc=12, n_restarts=6, seed=0):
        self.var_keep = var_keep; self.max_pc = max_pc
        self.n_restarts = n_restarts; self.seed = seed

    def fit(self, theta, G):
        """theta: (N, d) transformed params.  G: (N, Npts) shape vectors."""
        self.gm = G.mean(0)
        Gc = G - self.gm
        U, S, Vt = np.linalg.svd(Gc, full_matrices=False)
        evr = (S ** 2) / np.sum(S ** 2)
        k = int(np.searchsorted(np.cumsum(evr), self.var_keep) + 1)
        k = max(2, min(k, self.max_pc, Vt.shape[0]))
        self.k = k
        self.comp = Vt[:k]                       # (k, Npts) principal directions
        Z = Gc @ self.comp.T                     # (N, k) scores
        self.gps = [GP(n_restarts=self.n_restarts, seed=self.seed + j).fit(theta, Z[:, j])
                    for j in range(k)]
        self.evr_kept = float(np.sum(evr[:k]))
        return self

    def predict_g(self, theta):
        theta = np.atleast_2d(theta)
        Z = np.column_stack([gp.predict(theta) for gp in self.gps])     # (M, k)
        return self.gm + Z @ self.comp                                   # (M, Npts)

    @staticmethod
    def chi2_profiled(g, sr_exp, sr_err):
        """chi^2 with the linear normalization sigma0/2 profiled out (and s0/2)."""
        w = 1.0 / sr_err ** 2
        A = np.sum(g * g * w); B = np.sum(g * sr_exp * w); D = np.sum(sr_exp * sr_exp * w)
        if A <= 0 or not np.isfinite(A):
            return np.inf, np.nan
        return D - B * B / A, B / A

    def predict_chi2(self, theta, sr_exp, sr_err):
        """Return (chi2, s0half) arrays for a batch of theta."""
        G = self.predict_g(theta)
        out = [self.chi2_profiled(g, sr_exp, sr_err) for g in G]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])


# =============================================================================
#  Regional dispatcher: blend sub-emulators along one input axis
# =============================================================================
class RegionalEmulator:
    """Soft dispatch between sub-emulators along one input coordinate.

    Built for parameter boxes where a single stationary GP is limited by
    nonstationarity (e.g. the MV TMD map varies much faster at low C^2): train
    one ShapeEmulator per region (with overlapping training ranges) and
    cross-fade their predictions linearly over a small blend band at each
    interior boundary, so the combined prediction is continuous.

    emus     : list of trained ShapeEmulator, ordered along the axis
    edges    : interior boundaries (len = len(emus) - 1), in the axis coordinate
    axis     : input column used for dispatch
    blend    : half-width of the linear cross-fade around each edge
    Exposes predict_g(X) with the same signature/output as ShapeEmulator.
    """

    def __init__(self, emus, edges, axis=1, blend=0.15):
        assert len(edges) == len(emus) - 1
        self.emus, self.edges, self.axis, self.blend = emus, list(edges), axis, blend

    def _weights(self, x):
        """Per-region weights for scalar coordinate x (piecewise-linear fade)."""
        w = np.zeros(len(self.emus))
        # region index by edges
        for i in range(len(self.emus)):
            lo = -np.inf if i == 0 else self.edges[i - 1]
            hi = np.inf if i == len(self.emus) - 1 else self.edges[i]
            if lo - self.blend < x < hi + self.blend:
                wl = np.clip((x - (lo - self.blend)) / (2 * self.blend), 0, 1) if np.isfinite(lo) else 1.0
                wh = np.clip(((hi + self.blend) - x) / (2 * self.blend), 0, 1) if np.isfinite(hi) else 1.0
                w[i] = min(wl, wh)
        return w / max(w.sum(), 1e-300)

    def predict_g(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = None
        for irow, row in enumerate(X):
            w = self._weights(row[self.axis])
            pred = None
            for wi, emu in zip(w, self.emus):
                if wi <= 0:
                    continue
                p = emu.predict_g(row[None, :])[0]
                pred = wi * p if pred is None else pred + wi * p
            if out is None:
                out = np.empty((len(X), len(pred)))
            out[irow] = pred
        return out


class ColumnStitchEmulator:
    """Stitch sub-emulators that each predict a SUBSET of the output columns
    (e.g. separate Y-blocks of a flattened (Y, TMD, kT) grid), with per-column
    weights that cross-fade over shared overlap columns.

    emus    : list of sub-emulators (each with predict_g -> (M, len(cols_i)))
    cols    : list of integer index arrays -- target columns of each sub-emulator
    weights : list of float arrays (same shapes as cols); must sum to 1 per column
    ncols   : total output length
    Exposes predict_g(X) with the full-length output.
    """

    def __init__(self, emus, cols, weights, ncols):
        self.emus, self.cols, self.weights, self.ncols = emus, cols, weights, ncols

    def predict_g(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros((len(X), self.ncols))
        for emu, idx, w in zip(self.emus, self.cols, self.weights):
            out[:, idx] += w[None, :] * emu.predict_g(X)
        return out
