# Small-x gluon TMD explorer

Interactive [Streamlit](https://streamlit.io) app: turn the (Qs0², C²) knobs and
watch all seven small-x gluon TMDs (Appendix A of arXiv:2512.21466 +
Weizsäcker–Williams) respond live, at any rapidity Y = 0..10.

The theory is evaluated by a **Gaussian-process emulator** of the full
running-coupling BK + TMD pipeline. Since **v6** the targets come from the
**analytic-model pipeline** (`smallx_tmds` production default): the effective
anomalous dimension φ = d ln Γ/d ln r = 2γ_eff is fitted once per dipole slice
and *every* TMD is built from analytic derivatives of the fit — no smoothing,
no masks, no tail splices for A1–A6. Trained on 224 pipeline runs — one full
(kT, Y) TMD grid in ~1 ms instead of ~40 s.

## Validity box & accuracy
- Qs0² ∈ [0.05, 1.0] GeV², C² ∈ [0.5, 30]   (MV model: γ = e_c = 1)
- kT ∈ [0.01, 100] GeV (80 points), Y ∈ [0, 10] (ΔY = 0.2)
- held-out accuracy (v6 analytic-pipeline regional emulator, 224 training
  runs): median |rel. error| ≈ **0.08%, 95th pct 1.1%** — a ~2× (median) and
  3× (tail) improvement over v5, because the analytic targets carry no splice
  kinks or deep-UV quadrature noise. Predictions agree with v5 to ≤0.23%
  median across the box. The display applies a light smoothing along kT (the
  pickled emulator is raw — see TMD_emulator.pdf, Sec. 6).

## Methods
The emulator setup, validation, and an explicit account of every introduced
artifact are documented in [`TMD_emulator.pdf`](TMD_emulator.pdf); the
underlying TMD numerics in the companion repo `smallx_tmds`
(`docs/TMD_numerics.pdf`).  Sister app for the HERA reduced cross-section fit:
<https://dis-cgc-lo-rcbk-hera.streamlit.app>.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

- `app.py` — the Streamlit app
- `mv_tmd_emulator.pkl` — the trained emulator (from
  `smallx_tmds/scripts/build_mv_tmd_emulator.py`)
- `emulator.py` — GP/PCA classes needed to unpickle it
