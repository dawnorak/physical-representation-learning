"""
Plot train / val loss curves from losses.txt (two PNGs).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_losses_txt(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    assign_re = re.compile(
        r"^([a-zA-Z_][a-zA-Z0-9_.]*)_(train|val)\s*=\s*\[\s*$"
    )
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            i += 1
            continue
        m = assign_re.match(line)
        if not m:
            i += 1
            continue
        key = f"{m.group(1)}_{m.group(2)}"
        buf: list[float] = []
        i += 1
        while i < len(lines):
            ln = lines[i]
            if re.match(r"^\]\s*$", ln):
                break
            for s in FLOAT_RE.findall(ln):
                buf.append(float(s))
            i += 1
        out[key] = buf
        i += 1
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    txt = root / "losses.txt"
    data = parse_losses_txt(txt)

    # Display order (legend / z-order same): train keys and val keys per run.
    runs: list[tuple[str, str, str]] = [
        ("ConvSmall Default (N1)", "conv_small_n1_train", "conv_small_n1_val"),
        ("ConvLarge Default (N1)", "conv_large_default_n1_train", "conv_large_default_n1_val"),
        ("ConvLarge N0", "conv_large_n0_train", "conv_large_n0_val"),
        (
            "ConvLarge N0.5",
            "conv_large_n05_train",
            "conv_large_n0.5_val",
        ),
        (
            "ConvLarge Physics",
            "conv_large_physics_n1_train",
            "conv_large_physics_n1_val",
        ),
        (
            "ConvSmall Temporal",
            "conv_small_temporal_n1_train",
            "conv_small_temporal_n1_val",
        ),
        ("ConvSmall 2+1D", "conv_small_2plus1d_n1_train", "conv_small_2plus1d_n1_val"),
        (
            "ConvLarge 2+1D",
            "conv_large_2plus1d_n1_train",
            "conv_large_2plus1d_n1_val",
        ),
    ]

    colors = [
        "#0173B2",
        "#DE8F05",
        "#029E73",
        "#CC78BC",
        "#CA9161",
        "#949494",
        "#ECE133",
        "#56B4E9",
    ]

    for label, tk, vk in runs:
        if tk not in data or vk not in data:
            raise KeyError(f"Missing keys for {label}: {tk=} {vk=} keys={sorted(data)}")

    def plot_split(suffix: str, key_idx: int) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        for j, (label, tk, vk) in enumerate(runs):
            series = data[tk if key_idx == 0 else vk]
            x = list(range(len(series)))
            ax.plot(
                x,
                series,
                color=colors[j % len(colors)],
                label=label,
                linewidth=2,
                alpha=0.92,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        title = "Training loss" if key_idx == 0 else "Validation loss"
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        out = root / f"loss_curves_{suffix}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(out)

    plot_split("train", 0)
    plot_split("val", 1)


if __name__ == "__main__":
    main()
