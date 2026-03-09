# SRFM Paper — Reproducible Implementation

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b)](https://github.com/Mattbusel/srfm-paper-impl/blob/main/paper/srfm_paper.pdf)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/Mattbusel/srfm-python)
[![Rust](https://img.shields.io/badge/rust-1.77%2B-orange)](rust/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Reproducible research repository for:

> **Special-Relativistic Finance Manifold (SRFM): An Operational C++20 Implementation**
> Matthew C. Busel — 2025

The paper maps special-relativistic geometry onto OHLCV bar data: price velocity β,
Lorentz factor γ, and spacetime interval s² partition market regimes into TIMELIKE
and SPACELIKE sectors.  Q1 2025 empirical validation shows a variance ratio
**VR = 1.27×** between regimes, with Bartlett p = 6 × 10⁻¹⁶.

---

## Quick Start

```bash
git clone https://github.com/Mattbusel/srfm-paper-impl
cd srfm-paper-impl

# Install Python dependencies
pip install -e ".[all]"

# Reproduce all 8 figures (synthetic data, no market subscription needed)
python generate_all.py

# Or open the interactive notebook
jupyter notebook notebooks/reproduce_figures.ipynb
```

### Rust reference implementation

```bash
cd rust
cargo test   # 10 unit tests
cargo run    # print synthetic bar table
```

---

## Repository Structure

```
srfm-paper-impl/
├── paper/
│   └── srfm_paper.pdf          # Full paper (27 pages)
│
├── notebooks/
│   └── reproduce_figures.ipynb # Jupyter notebook — all 8 figures + live stats
│
├── src/figures/
│   ├── fig1_spacetime_diagram.py        # Fig 1 — Minkowski diagram
│   ├── fig2_lorentz_factor_surface.py   # Fig 2 — γ(β), time dilation, rapidity
│   ├── fig3_regime_distributions.py     # Fig 3 — return distributions by regime
│   ├── fig4_variance_ratio_heatmap.py   # Fig 4 — VR hyperparameter sensitivity
│   ├── fig5_geodesic_deviation.py       # Fig 5 — geodesic deviation timeseries
│   ├── fig6_cumulative_pnl.py           # Fig 6 — strategy P&L vs buy-and-hold
│   ├── fig7_covariance_manifold.py      # Fig 7 — SPD manifold trajectory
│   └── fig8_module_pipeline.py          # Fig 8 — pipeline data-flow diagram
│
├── rust/
│   ├── Cargo.toml
│   └── src/main.rs                      # Pure-Rust β, γ, s², τ, φ implementation
│
├── data/figures/                        # Generated PDFs (created by generate_all.py)
│
├── generate_all.py                      # Reproduce every figure in one command
├── pyproject.toml
└── requirements.txt
```

---

## Key Results

| Metric | Value |
|---|---|
| Variance ratio VR (99.5th pctile, n=20) | **1.27×** |
| Bartlett test p-value | **6 × 10⁻¹⁶** |
| TIMELIKE fraction of 1-min bars | ~75% |
| SRFM strategy Sharpe ratio (Q1 2025) | ~1.8 |
| Buy-and-hold Sharpe ratio (Q1 2025) | ~1.1 |
| Pipeline P50 latency (Beta calc) | 48 ns |

---

## Related Repositories

| Repo | Description |
|---|---|
| [srfm-python](https://github.com/Mattbusel/srfm-python) | Python SDK — `pip install srfm` |
| [Special-Relativity-in-Financial-Modeling](https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling) | C++20 production implementation |
| [agent-runtime](https://github.com/Mattbusel/agent-runtime) | Tokio async runtime used in the CI/CD pipeline |

---

## Citing

```bibtex
@article{busel2025srfm,
  title   = {Special-Relativistic Finance Manifold: An Operational {C++20} Implementation},
  author  = {Busel, Matthew C.},
  year    = {2025},
  url     = {https://github.com/Mattbusel/srfm-paper-impl}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
