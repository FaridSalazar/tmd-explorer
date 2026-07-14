#!/usr/bin/env python3
# =============================================================================
#  Interactive small-x gluon TMD explorer -- MV model (gamma = e_c = 1).
#  Turn the (Qs0^2, C^2) knobs; a Gaussian-process emulator of the full
#  rcBK + TMD pipeline (S-matrix evolution, IBP A1-A6, K-fit WW) returns all
#  seven TMDs on an 80-point kT grid at 51 rapidities Y = 0..10 in ~1 ms.
#
#  Emulator validity box: Qs0^2 in [0.05, 1.0] GeV^2,  C^2 in [0.5, 30].
#  v3 regional emulator (C^2 split at 4, blended): held-out median ~0.1%,
#  sub-percent at all probe points incl. the low-C^2 corner.
#
#  Run:   streamlit run app.py
# =============================================================================
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                     # emulator.py (needed to unpickle)

st.set_page_config(page_title="small-x gluon TMD explorer", layout="wide")

BOX = dict(Qs02=(0.05, 1.0), C2=(0.5, 30.0))
BEST = dict(Qs02=0.104, C2=15.06)            # MV fit to HERA 2009 (x<0.01)

LABELS = {"qg1": r"${\mathcal{F}}^{(1)}_{qg}$", "qg2": r"${\mathcal{F}}^{(2)}_{qg}$",
          "gg1": r"${\mathcal{F}}^{(1)}_{gg}$", "gg2": r"${\mathcal{F}}^{(2)}_{gg}$",
          "gg3": r"${\mathcal{F}}^{(3)}_{gg}$", "adj": r"${\mathcal{F}}_{\mathrm{Adj}}$",
          "ww": r"${\mathcal{F}}_{\mathrm{WW}}$"}
COLORS = {"qg1": "C0", "qg2": "C1", "gg1": "C2", "gg3": "C3",
          "adj": "C4", "gg2": "C5", "ww": "k"}


@st.cache_resource(show_spinner="Loading the trained TMD emulator...")
def load_bundle():
    with open(os.path.join(HERE, "mv_tmd_emulator.pkl"), "rb") as f:
        return pickle.load(f)


B = load_bundle()
KT, YS, KEYS = B["kT"], np.asarray(B["YS"]), B["keys"]


@st.cache_data(max_entries=64)
def predict_grid(qs02, c2):
    """All seven TMDs on the full (Y, kT) grid: F[key][iY, ikT].

    A light Savitzky-Golay filter along ln kT (w=9, p=3) is applied FOR DISPLAY:
    the direct pipeline curves are verified smooth at grid scale, so this removes
    residual GP ringing (visible at low Y) without visibly biasing the curves.
    """
    v = B["emu"].predict_g([[np.log(qs02), np.log(c2)]])[0]
    v = savgol_filter(v.reshape(len(YS), len(KEYS), len(KT)), 9, 3, axis=2)
    A = np.exp(v)
    # mask the training-floor region (values < ~peak - 5.2 decades are an
    # emulator conditioning artifact, not physics -- curves end instead of
    # shelving; see TMD_emulator.pdf Sec. 4)
    peak = A.max(axis=2, keepdims=True)
    A = np.where(A > peak * 10**-5.3, A, np.nan)
    F = {k: A[:, i, :] for i, k in enumerate(KEYS)}
    F["gg2"] = F["gg1"] - F["adj"]            # exact reconstruction
    return F


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Small-x gluon TMD explorer  (MV dipole)")
st.caption(
    "Move the knobs; a Gaussian-process emulator of the running-coupling BK + "
    "TMD pipeline (Appendix A of arXiv:2512.21466 + Weizsäcker–Williams) returns "
    "all seven gluon TMDs on the full $(k_T, Y)$ grid instantly.  "
    "Sliders span the emulator's validity box."
)

cL, cR = st.columns([1, 3])
with cL:
    st.subheader("Dipole parameters")
    qs02 = st.slider(r"$Q_{s0}^{2}$ [GeV$^{2}$]", *BOX["Qs02"], BEST["Qs02"], 0.005,
                     key="qs02", help="Initial saturation scale squared at x0 = 0.01")
    c2 = st.slider(r"$C^{2}$", *BOX["C2"], BEST["C2"], 0.25,
                   key="c2", help="Running-coupling scale constant (evolution speed)")
    st.subheader("View")
    yy = st.slider("rapidity  Y = ln(x₀/x)", float(YS[0]), float(YS[-1]),
                   2.0, float(YS[1] - YS[0]))
    st.caption(rf"x = {0.01 * np.exp(-yy):.2e}")
    shown = st.multiselect("TMDs", ["qg1", "qg2", "gg1", "gg2", "gg3", "adj", "ww"],
                           default=["qg1", "qg2", "gg1", "gg3", "adj", "ww"])
    ref = st.checkbox("show MV HERA best fit (grey)", value=True)
    kfix = st.slider(r"$k_{T}$ for the evolution panel [GeV]",
                     0.5, 10.0, 1.0, 0.5)
    def _reset_to_hera():
        st.session_state["qs02"] = BEST["Qs02"]
        st.session_state["c2"] = BEST["C2"]

    st.button("reset to HERA best fit", on_click=_reset_to_hera)

F = predict_grid(qs02, c2)
Fbase = predict_grid(BEST["Qs02"], BEST["C2"])       # HERA baseline (always, for ratios)
Fref = Fbase if ref else None
jy = int(np.argmin(np.abs(YS - yy)))
ik = int(np.argmin(np.abs(KT - kfix)))

with cR:
    fig, (ax, axr, axy) = plt.subplots(3, 1, figsize=(8.2, 12.0),
                                       gridspec_kw=dict(height_ratios=[1.3, 1, 1],
                                                        hspace=0.35))
    qsnorm = qs02 / BEST["Qs02"]                 # divide out the trivial Qs0^2 scaling
    for k in shown:
        ax.plot(KT, np.abs(F[k][jy]), color=COLORS[k], lw=2.4, label=LABELS[k])
        if Fref is not None:
            ax.plot(KT, np.abs(Fref[k][jy]), color="0.65", lw=1.0, zorder=0)
        axr.plot(KT, F[k][jy] / Fbase[k][jy] / qsnorm, color=COLORS[k], lw=2.0)
        axy.plot(YS, np.abs(F[k][:, ik]), color=COLORS[k], lw=2.2)
        if Fref is not None:
            axy.plot(YS, np.abs(Fref[k][:, ik]), color="0.65", lw=1.0, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(KT[0], KT[-1])
    # dynamic lower limit: show the full high-kT tails of ALL plotted curves
    # (current parameters and, when shown, the grey HERA baseline)
    lo = np.nanmin([np.nanmin(np.abs(F[k][jy])) for k in shown]
                + ([np.nanmin(np.abs(Fbase[k][jy])) for k in shown] if Fref is not None else []))
    ax.set_ylim(max(0.5 * lo, 1e-9), 1e0)
    ax.set_xlabel(r"$k_{T}$ [GeV]")
    ax.set_ylabel(r"$\alpha_{s}\,{\mathcal{F}}^{(i)}(k_{T})/S_{\perp}$")
    ax.set_title(rf"TMDs at $Y={yy:g}$   (x = {0.01*np.exp(-yy):.1e})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left")
    axr.axhline(1.0, color="0.4", lw=0.8, ls="--")
    axr.set_xscale("log"); axr.set_xlim(KT[0], KT[-1])
    axr.set_xlabel(r"$k_{T}$ [GeV]")
    axr.set_ylabel(r"$\dfrac{{\mathcal{F}}/{\mathcal{F}}_{\mathrm{HERA}}}"
                   r"{Q_{s0}^{2}/Q_{s0,\mathrm{HERA}}^{2}}$")
    axr.set_title(rf"ratio to HERA fit at $Y={yy:g}$, "
                  rf"normalized by $Q_{{s0}}^{{2}}$ ratio ($={qsnorm:.2f}$)")
    axr.grid(alpha=0.3, which="both")
    axy.axvline(yy, color="0.5", ls=":", lw=1)
    axy.set_yscale("log")
    axy.set_xlabel(r"$Y$"); axy.set_title(rf"evolution at $k_{{T}}={kfix}$ GeV")
    axy.grid(alpha=0.3, which="both")
    try:
        fig.tight_layout()
    except Exception:
        pass
    st.pyplot(fig, clear_figure=True)

    st.caption(
        rf"$Q_{{s0}}^2 = {qs02:.3f}$ GeV$^2$, $C^2 = {c2:.2f}$  |  grey reference: "
        rf"MV HERA fit ($Q_{{s0}}^2={BEST['Qs02']}$, $C^2={BEST['C2']}$)  |  "
        "emulator v3 (regional GP, 224 training BK runs): median accuracy ~0.1%; "
        "display uses light smoothing along $k_{T}$ (raw emulator in the pickle)."
    )
