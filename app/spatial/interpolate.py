"""Spatial interpolation of FGPI point samples onto a regular grid.

Default method is inverse-distance weighting (IDW, power 2). Cubic uses
scipy.interpolate.griddata when scipy is installed and enough well-spread
points exist; otherwise it silently falls back to IDW. Both return a
(resolution x resolution) float array, row 0 = y 0.
"""

from __future__ import annotations

import numpy as np

from app import load_config

_IDW_POWER = 2.0
_EPS = 1e-9


def _idw(
    px: np.ndarray, py: np.ndarray, pv: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    resolution = xs.size
    grid = np.empty((resolution, resolution), dtype=float)
    # Chunk rows so large point sets do not allocate a huge 3-D array.
    chunk = max(1, min(25, resolution))
    for start in range(0, resolution, chunk):
        stop = min(resolution, start + chunk)
        X, Y = np.meshgrid(xs, ys[start:stop])
        dx = X[..., None] - px
        dy = Y[..., None] - py
        dist2 = dx * dx + dy * dy
        weights = 1.0 / np.power(dist2 + _EPS, _IDW_POWER / 2.0)
        values = np.sum(weights * pv, axis=-1) / np.sum(weights, axis=-1)
        # Exact hits should reproduce the sample, not an epsilon-weighted blend.
        exact = dist2 <= _EPS
        if np.any(exact):
            hit = np.argmax(exact, axis=-1)
            values = np.where(np.any(exact, axis=-1), pv[hit], values)
        grid[start:stop, :] = values
    return grid


def _cubic(
    px: np.ndarray, py: np.ndarray, pv: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray | None:
    try:
        from scipy.interpolate import griddata  # type: ignore
    except ImportError:
        return None
    if px.size < 4:
        return None
    X, Y = np.meshgrid(xs, ys)
    try:
        cubic = griddata((px, py), pv, (X, Y), method="cubic")
        if cubic is None or not np.any(np.isfinite(cubic)):
            return None
        if not np.all(np.isfinite(cubic)):
            nearest = griddata((px, py), pv, (X, Y), method="nearest")
            cubic = np.where(np.isfinite(cubic), cubic, nearest)
        return cubic
    except Exception:
        return None


def grid_data(
    points: list[tuple[float, float, float]],
    x_max: float,
    y_max: float,
    resolution: int = 100,
) -> np.ndarray:
    """Interpolate (x, y, value) samples over [0, x_max] x [0, y_max]."""
    resolution = max(2, int(resolution))
    x_max = max(float(x_max), 1e-6)
    y_max = max(float(y_max), 1e-6)
    xs = np.linspace(0.0, x_max, resolution)
    ys = np.linspace(0.0, y_max, resolution)

    clean: list[tuple[float, float, float]] = []
    for point in points or []:
        try:
            x, y, value = float(point[0]), float(point[1]), float(point[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(value)):
            continue
        clean.append((x, y, value))

    if not clean:
        return np.zeros((resolution, resolution), dtype=float)

    # Keep the newest sample for coincident points (last wins) so repeated
    # fixed-node readings do not overweight one location.
    latest: dict[tuple[float, float], float] = {}
    for x, y, value in clean[-5000:]:
        latest[(x, y)] = value
    arr = np.asarray([(x, y, v) for (x, y), v in latest.items()], dtype=float)
    px, py, pv = arr[:, 0], arr[:, 1], arr[:, 2]

    if px.size == 1:
        return np.full((resolution, resolution), float(pv[0]), dtype=float)

    method = str(load_config().get("spatial", {}).get("method", "idw")).lower()
    grid = _cubic(px, py, pv, xs, ys) if method == "cubic" else None
    if grid is None:
        grid = _idw(px, py, pv, xs, ys)

    return np.clip(np.asarray(grid, dtype=float), 0.0, 100.0)
