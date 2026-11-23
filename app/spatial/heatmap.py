"""Matplotlib heatmap rendering.

Keeps rendering under ~3 s on a Pi 4 by staying at 110 dpi, using Agg, and
writing atomically (temp file + os.replace) so Flask never serves a half-
written PNG while the sampler refreshes data.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from app import load_config

# Green -> amber -> red ramp; readable in greyscale-ish print and colour-blind
# friendly enough for a risk surface (not a substitute for accessible text).
_COLORS = ["#e2e8f0", "#99f6e4", "#fde68a", "#fb923c", "#dc2626"]


def _optimize_png(path: str) -> None:
    """Downsample + palette-quantise the rendered PNG.

    The map is a five-stop risk surface, so 48 colours are plenty. This keeps
    the transferred heatmap (and the browser's total asset budget) small even
    when the interpolated field has lots of detail.
    """
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.width > 560:
            height = max(1, round(rgb.height * 560 / rgb.width))
            rgb = rgb.resize((560, height), Image.Resampling.LANCZOS)
        quantised = rgb.quantize(
            colors=64, method=Image.MEDIANCUT, dither=Image.Dither.NONE
        )
        quantised.save(path, format="PNG", optimize=True)


def render_png(grid: np.ndarray, out_path: str) -> None:
    """Save a heatmap PNG with baked-in axes, title and FGPI legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("grid must be a non-empty 2-D array")
    arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=0.0)

    spatial = load_config().get("spatial", {})
    x_max = float(spatial.get("x_max", 10.0))
    y_max = float(spatial.get("y_max", 10.0))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list("fgpi_risk", _COLORS)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=110)
    image = ax.imshow(
        arr,
        origin="lower",
        extent=(0.0, x_max, 0.0, y_max),
        vmin=0.0,
        vmax=100.0,
        cmap=cmap,
        interpolation="bilinear",
        aspect="auto",
    )
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("FGPI spatial interpolation (predicted, not detected)", fontsize=10)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("FGPI (0-100)")

    fd, tmp_name = tempfile.mkstemp(prefix="heatmap-", suffix=".png", dir=str(out.parent))
    os.close(fd)
    try:
        fig.savefig(
            tmp_name,
            format="png",
            dpi=110,
            bbox_inches="tight",
            metadata={"Software": "mushy-fungi-forecast"},
            pil_kwargs={"compress_level": 9},
        )
        _optimize_png(tmp_name)
        os.replace(tmp_name, out)
    finally:
        plt.close(fig)
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
