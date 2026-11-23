"""Matplotlib map rendering for every dashboard layer.

Keeps rendering under ~2 s on a Pi 4 by staying at 110 dpi, using Agg, and
writing atomically (temp file + os.replace) so Flask never serves a half-
written PNG while the sampler refreshes data.

`render_png` keeps its spec signature (FGPI only). `render_layer_png` draws any
registered layer and, when `sites` is supplied, labels each sensor site with its
node id, FGPI score and dominant predicted species.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from app import load_config
from app.spatial.layers import Layer, get_layer


def _optimize_png(path: str) -> None:
    """Downsample + palette-quantise the rendered PNG.

    The maps are five-to-six stop surfaces, so 64 colours are plenty. This keeps
    the transferred PNG small even when the interpolated field has lots of
    detail; antialiased annotation text survives because its greys survive the
    palette.
    """
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.width > 640:
            height = max(1, round(rgb.height * 640 / rgb.width))
            rgb = rgb.resize((640, height), Image.Resampling.LANCZOS)
        quantised = rgb.quantize(
            colors=64, method=Image.MEDIANCUT, dither=Image.Dither.NONE
        )
        quantised.save(path, format="PNG", optimize=True)


def _domain() -> tuple[float, float]:
    spatial = load_config().get("spatial", {})
    return float(spatial.get("x_max", 10.0)), float(spatial.get("y_max", 8.0))


def _save(fig, out: Path) -> None:
    import matplotlib.pyplot as plt

    fd, tmp_name = tempfile.mkstemp(prefix="map-", suffix=".png", dir=str(out.parent))
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


def render_png(grid: np.ndarray, out_path: str) -> None:
    """Save the FGPI heatmap PNG with baked-in axes, title and legend."""
    render_layer_png(grid, out_path, layer="fgpi")


def render_layer_png(
    grid: np.ndarray,
    out_path: str,
    layer: str | Layer = "fgpi",
    sites: list[dict] | None = None,
    x_max: float | None = None,
    y_max: float | None = None,
) -> None:
    """Save one layer's map. Each entry in `sites` needs x, y and label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    spec = layer if isinstance(layer, Layer) else get_layer(layer)
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("grid must be a non-empty 2-D array")
    arr = np.nan_to_num(arr, nan=spec.lo, posinf=spec.hi, neginf=spec.lo)
    arr = np.clip(arr, spec.lo, spec.hi)
    domain_x, domain_y = _domain()
    x_span = float(x_max if x_max is not None else domain_x)
    y_span = float(y_max if y_max is not None else domain_y)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list(f"{spec.key}_map", list(spec.colors))
    fig, ax = plt.subplots(figsize=(6.6, 4.7), dpi=110)
    image = ax.imshow(
        arr,
        origin="lower",
        extent=(0.0, x_span, 0.0, y_span),
        vmin=spec.lo,
        vmax=spec.hi,
        cmap=cmap,
        interpolation="bilinear",
        aspect="auto",
    )
    ax.set_xlim(0.0, x_span)
    ax.set_ylim(0.0, y_span)
    ax.set_xlabel("x position (m)", fontsize=8)
    ax.set_ylabel("y position (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"{spec.label} over the monitored area - modelled, not detected",
        fontsize=9.5,
        pad=8,
    )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(spec.unit, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    if spec.log_axis:
        cbar.set_ticks(list(spec.ticks))
        cbar.set_ticklabels([spec.inverse(t) for t in spec.ticks])
    elif spec.lo <= 0.0 and spec.hi <= 100.5:
        cbar.set_ticks([t for t in spec.ticks if spec.lo <= t <= spec.hi])

    if sites:
        _annotate_sites(ax, fig, sites)

    _save(fig, out)


# Offsets tried, in order, before a label is dropped: right, left, above,
# below, and the two upper diagonals. Keeps dense sensor sets readable.
_LABEL_OFFSETS = ((5, 4), (-5, 4), (5, -8), (-5, -8), (5, 11), (-5, 11))


def _annotate_sites(ax, fig, sites: list[dict]) -> None:
    """Draw site dots and place labels without overlapping each other.

    Greedy placement: for each site try the candidate offsets and keep the first
    box that does not collide with an already-placed label or a previously
    placed dot. Sites that cannot be placed keep their dot (so the sample
    location is never lost) and are counted; the caller-facing title stays
    honest about it.
    """
    ax.scatter(
        [float(s["x"]) for s in sites],
        [float(s["y"]) for s in sites],
        s=24,
        facecolor="#ffffff",
        edgecolor="#0f172a",
        linewidth=0.9,
        zorder=3,
    )
    renderer = fig.canvas.get_renderer()
    placed: list = []
    skipped = 0
    for site in sites:
        x, y = float(site["x"]), float(site["y"])
        for dx, dy in _LABEL_OFFSETS:
            annotation = ax.annotate(
                site["label"],
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.0,
                color="#0f172a",
                zorder=4,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.85,
                    "linewidth": 0.5,
                },
            )
            box = annotation.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
            annotation.remove()
        else:
            skipped += 1
    if skipped:
        ax.set_xlabel(
            f"x position (m) - {skipped} site label(s) hidden for legibility",
            fontsize=8,
        )