#!/usr/bin/env python
"""
generate_all.py — reproduce every figure in the SRFM paper.

Usage:
    python generate_all.py [--out-dir data/figures]

All figures use synthetic data by default so no market-data subscription
is required.  Pass --data-file to override with empirical .npz files.
"""

import argparse
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────


def ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── individual figure drivers ─────────────────────────────────────────────────


def fig1(out: Path) -> None:
    from src.figures.fig1_spacetime_diagram import make_figure
    make_figure(out / "fig1_spacetime_diagram.pdf")


def fig2(out: Path) -> None:
    from src.figures.fig2_lorentz_factor_surface import make_figure
    make_figure(out / "fig2_lorentz_factor_surface.pdf")


def fig3(out: Path) -> None:
    from src.figures.fig3_regime_distributions import (
        generate_synthetic_data, make_figure,
    )
    make_figure(generate_synthetic_data(), out / "fig3_regime_distributions.pdf")


def fig4(out: Path) -> None:
    from src.figures.fig4_variance_ratio_heatmap import make_heatmap
    make_heatmap(out / "fig4_variance_ratio_heatmap.pdf")


def fig5(out: Path) -> None:
    from src.figures.fig5_geodesic_deviation import (
        generate_synthetic_series, make_figure,
    )
    make_figure(generate_synthetic_series(), out / "fig5_geodesic_deviation.pdf")


def fig6(out: Path) -> None:
    from src.figures.fig6_cumulative_pnl import (
        generate_synthetic_series, make_figure,
    )
    make_figure(generate_synthetic_series(), out / "fig6_cumulative_pnl.pdf")


def fig7(out: Path) -> None:
    from src.figures.fig7_covariance_manifold import make_figure
    make_figure(out / "fig7_covariance_manifold.pdf")


def fig8(out: Path) -> None:
    from src.figures.fig8_module_pipeline import make_figure
    make_figure(out / "fig8_module_pipeline.pdf")


# ── main ──────────────────────────────────────────────────────────────────────

FIGURES = [
    ("Fig 1 — Minkowski diagram",         fig1),
    ("Fig 2 — Lorentz factor surface",    fig2),
    ("Fig 3 — Regime distributions",      fig3),
    ("Fig 4 — Variance ratio heatmap",    fig4),
    ("Fig 5 — Geodesic deviation series", fig5),
    ("Fig 6 — Cumulative P&L",            fig6),
    ("Fig 7 — Covariance manifold",       fig7),
    ("Fig 8 — Module pipeline",           fig8),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/figures",
                    help="Output directory for generated PDFs (default: data/figures)")
    args = ap.parse_args()

    out = ensure(Path(args.out_dir))
    print(f"Output directory: {out.resolve()}\n")

    for label, fn in FIGURES:
        print(f"  Generating {label}...", end=" ", flush=True)
        try:
            fn(out)
            print("ok")
        except Exception as exc:
            print(f"FAILED — {exc}")

    print("\nDone. PDFs saved to:", out.resolve())


if __name__ == "__main__":
    main()
