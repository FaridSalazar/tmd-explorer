# Small-x gluon TMD explorer

Interactive [Streamlit](https://streamlit.io) app: turn the dipole knobs and
watch all seven small-x gluon TMDs (Appendix A of arXiv:2512.21466 +
Weizsäcker–Williams) respond live, at any rapidity Y = 0..10.

Two pages (sidebar):

| page | dipole IC | parameters | emulator |
|---|---|---|---|
| **MV explorer** (default) | MV (γ = e_c = 1) | Qs0², C² | v7, 224 BK runs |
| **MVγ explorer** | MVγ | Qs0², C², γ | v2, 472 BK runs + exact Y=0 sub-emulator |

The theory is evaluated by **Gaussian-process emulators** of the full
running-coupling BK + TMD pipeline. The targets come from the
**analytic-model pipeline** (`smallx_tmds` production default): the effective
anomalous dimension φ = d ln Γ/d ln r = 2γ_eff is fitted once per dipole slice
and *every* TMD is built from analytic derivatives of the fit — no smoothing,
no masks, no tail splices for A1–A6; the WW is matched to F⁽²⁾_qg at high kT
with an adaptive splice. One full (kT, Y) TMD grid in ~1 ms instead of ~40 s.

## Validity box & accuracy
- **MV (v7)**: Qs0² ∈ [0.05, 1.0] GeV², C² ∈ [0.5, 30]; held-out median
  |rel. error| ≈ **0.08%, 95th pct 1.1%**.
- **MVγ (v2)**: same box + γ ∈ [1.0, 1.3]; held-out median ≈ **0.8%, 95th pct
  ≈ 9%** (the tail is concentrated in the moving γ>1 UV sign-crossing notch of
  the A1–A6 integrands). At Y = 0 an exact 2D (Qs0², γ) sub-emulator of the
  analytic initial condition is blended in over Y ≤ 0.6: the Y=0 slice error is
  0.9% median and is *exactly* C²-independent, as the physics requires.
- kT ∈ [0.01, 100] GeV (80 points), Y ∈ [0, 10] (ΔY = 0.2). The display applies
  a light smoothing along kT (the pickled emulators are raw — see
  TMD_emulator.pdf, Sec. 6).

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

- `app.py` — multipage entry point (`st.navigation`, MV is the default page)
- `mv_page.py` / `mvgamma_page.py` — the two explorer pages
- `mv_tmd_emulator.pkl` — MV v7 emulator (from
  `smallx_tmds/scripts/build_mv_tmd_emulator.py`)
- `mvgamma_tmd_emulator.pkl` — MVγ v2 emulator (from
  `smallx_tmds/scripts/build_mvgamma_tmd_emulator.py`; deployment copy with
  the GP Cholesky factors stripped — predictions are bit-identical, 308→26 MB)
- `emulator.py` — GP/PCA classes needed to unpickle them
