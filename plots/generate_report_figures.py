"""
Build Figures 1--3 for project_report_draft.tex from finetuning results/metrics.json
and feature_stats.json. Outputs raster PNG (not PDF). Run from repo root:

    python icml2026/generate_report_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_EXT = ".png"
FIG_DPI = 220

# Display order (must match project_report_draft.tex table and narrative).
RUNS: list[tuple[str, Path]] = [
    ("ConvSmall Default (N1)", Path("finetuning results/results/results_conv_small_default_n1")),
    ("ConvLarge Default (N1)", Path("finetuning results/results/convlarge_n1")),
    ("ConvLarge N0", Path("finetuning results/results/results_conv_large_n0")),
    ("ConvLarge N5", Path("finetuning results/results/results_conv_large_n5")),
    ("ConvLarge Physics", Path("finetuning results/results/results_conv_large_physics")),
    ("ConvSmall Temporal", Path("finetuning results/results/results_conv_small_temporal")),
    ("ConvSmall 2+1D", Path("finetuning results/results/convsmall_2p1d")),
    ("ConvLarge 2+1D", Path("finetuning results/results/convlarge_2p1d")),
]


def load_metrics(root: Path, folder: Path) -> dict:
    p = root / folder / "metrics.json"
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_feature_stats(root: Path, folder: Path) -> dict:
    p = root / folder / "feature_stats.json"
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def shorten_label(s: str) -> str:
    """Axis labels: shorten for readability."""
    return (
        s.replace("ConvSmall ", "Sm. ")
        .replace("ConvLarge ", "Lg. ")
        .replace(" Default ", " Def. ")
        .replace("Temporal", "Temp.")
        .replace("Physics", "Phys.")
    )


def fig1_main_probes(root: Path, out: Path) -> None:
    labels = [shorten_label(a) for a, _ in RUNS]
    n = len(labels)
    val_lin = np.zeros(n)
    te_lin = np.zeros(n)
    val_knn = np.zeros(n)
    te_knn = np.zeros(n)
    for i, (_, folder) in enumerate(RUNS):
        m = load_metrics(root, folder)
        val_lin[i] = m["linear"]["val"]["mse_mean"]
        te_lin[i] = m["linear"]["test"]["mse_mean"]
        val_knn[i] = m["knn"]["val"]["mse_mean"]
        te_knn[i] = m["knn"]["test"]["mse_mean"]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=150, sharey=True)
    y = np.arange(n)
    h = 0.36
    ax0.barh(y - h / 2, val_lin, h, label="Val", color="#0173B2", alpha=0.85)
    ax0.barh(y + h / 2, te_lin, h, label="Test", color="#DE8F05", alpha=0.85)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.set_xlabel(r"Mean MSE ($z$-scored $\alpha,\zeta$)")
    ax0.set_title("Linear probe")
    ax0.legend(loc="lower right", fontsize=8)
    ax0.grid(axis="x", alpha=0.35)

    ax1.barh(y - h / 2, val_knn, h, label="Val", color="#0173B2", alpha=0.85)
    ax1.barh(y + h / 2, te_knn, h, label="Test", color="#DE8F05", alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax1.set_xlabel(r"Mean MSE ($z$-scored $\alpha,\zeta$)")
    ax1.set_title(r"$k$NN probe ($k{=}5$, distance weights)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis="x", alpha=0.35)

    fig.suptitle("Frozen downstream error (lower is better)", fontsize=11, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=FIG_DPI, format="png")
    plt.close(fig)


def fig2_alpha_zeta(root: Path, out: Path) -> None:
    labels = [shorten_label(a) for a, _ in RUNS]
    n = len(labels)
    la = np.zeros(n)
    lz = np.zeros(n)
    ka = np.zeros(n)
    kz = np.zeros(n)
    for i, (_, folder) in enumerate(RUNS):
        m = load_metrics(root, folder)
        la[i] = m["linear"]["test"]["mse_alpha"]
        lz[i] = m["linear"]["test"]["mse_zeta"]
        ka[i] = m["knn"]["test"]["mse_alpha"]
        kz[i] = m["knn"]["test"]["mse_zeta"]

    plt.rcParams.update({"font.size": 9})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=150, sharey=True)
    y = np.arange(n)
    h = 0.36
    ax0.barh(y - h / 2, la, h, label=r"$\alpha$", color="#029E73", alpha=0.9)
    ax0.barh(y + h / 2, lz, h, label=r"$\zeta$", color="#CC78BC", alpha=0.9)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.set_xlabel(r"Test MSE (per parameter)")
    ax0.set_title("Linear probe (test)")
    ax0.legend(loc="lower right", fontsize=8)
    ax0.grid(axis="x", alpha=0.35)

    ax1.barh(y - h / 2, ka, h, label=r"$\alpha$", color="#029E73", alpha=0.9)
    ax1.barh(y + h / 2, kz, h, label=r"$\zeta$", color="#CC78BC", alpha=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax1.set_xlabel(r"Test MSE (per parameter)")
    ax1.set_title(r"$k$NN probe (test)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis="x", alpha=0.35)

    fig.suptitle(r"$\alpha$ vs.\ $\zeta$: $\zeta$ is systematically harder", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=FIG_DPI, format="png")
    plt.close(fig)


def fig3_feature_stats(root: Path, out: Path) -> None:
    labels = [shorten_label(a) for a, _ in RUNS]
    n = len(labels)
    mean_var = np.zeros(n)
    s1 = np.zeros(n)
    for i, (_, folder) in enumerate(RUNS):
        fs = load_feature_stats(root, folder)
        mean_var[i] = fs["train"]["mean_feature_variance"]
        s1[i] = fs["train"]["top_singular_values"][0]

    plt.rcParams.update({"font.size": 9})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.8), dpi=150)
    y = np.arange(n)
    ax0.barh(y, mean_var, color="#CA9161", alpha=0.9)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.set_xlabel("Mean feature variance (train features)")
    ax0.set_title("Feature spread")
    ax0.grid(axis="x", alpha=0.35)

    ax1.barh(y, s1, color="#56B4E9", alpha=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax1.set_xlabel("Largest singular value (train features)")
    ax1.set_title("Top singular value")
    ax1.grid(axis="x", alpha=0.35)

    fig.suptitle("Representation diagnostics (no collapse in evaluated runs)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=FIG_DPI, format="png")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(__file__).resolve().parent / "figures"
    fig1_main_probes(root, out_dir / f"fig1_frozen_probe_main{FIG_EXT}")
    fig2_alpha_zeta(root, out_dir / f"fig2_alpha_zeta{FIG_EXT}")
    fig3_feature_stats(root, out_dir / f"fig3_feature_stats{FIG_EXT}")
    for name in ("fig1_frozen_probe_main", "fig2_alpha_zeta", "fig3_feature_stats"):
        print("Wrote:", out_dir / f"{name}{FIG_EXT}")


if __name__ == "__main__":
    main()
