"""Map layer registry: one spec per quantity the dashboard can draw.

Each layer declares its units, colour ramp, value transform and how to format
readings. The server uses this to interpolate + render; the template uses it to
build the dropdown, so adding a layer is a single entry here.

Lux is interpolated in log space (log10(1 + lux)) because indoor light spans
orders of magnitude and a linear ramp washes out anything below ~200 lux.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Layer:
    key: str
    label: str
    heading: str
    unit: str
    colors: tuple[str, ...]
    ticks: tuple[float, ...]
    transform: Callable[[float], float] = field(default=lambda v: v)
    inverse: Callable[[float], float] = field(default=lambda v: v)
    fmt: str = "{:.0f}"
    note: str = ""
    # Colour-ramp bounds in transformed space; None = fixed domain.
    lo: float = 0.0
    hi: float = 100.0
    log_axis: bool = False


def _log_lux(value):
    return np.log10(1.0 + np.maximum(0.0, value))


def _lux_label(value: float) -> str:
    lux = 10.0 ** float(value) - 1.0
    if lux >= 1000:
        return f"{lux / 1000:.1f}k"
    if lux >= 10:
        return f"{lux:.0f}"
    # Negative or unreadable sensor (read_lux() -> -1.0) shows as "n/a".
    return "n/a" if lux < 0.5 else f"{lux:.1f}"


LAYERS: dict[str, Layer] = {
    "fgpi": Layer(
        key="fgpi",
        label="FGPI",
        heading="FGPI map",
        unit="FGPI (0-100)",
        colors=("#e2e8f0", "#99f6e4", "#fde68a", "#fb923c", "#dc2626"),
        ticks=(0, 25, 50, 75, 100),
        hi=100.0,
        note="Green = low, amber = moderate/high, red = critical.",
    ),
    "temperature": Layer(
        key="temperature",
        label="Temperature",
        heading="Temperature map",
        unit="temperature (C)",
        colors=("#1d4ed8", "#0ea5e9", "#a3e635", "#facc15", "#f97316", "#dc2626"),
        ticks=(0, 10, 20, 30, 40),
        lo=0.0,
        hi=40.0,
        fmt="{:.1f}",
        note="Cold-tolerant moulds (e.g. Eurotium) still grow near 10 C; growth is optimal 20-25 C.",
    ),
    "humidity": Layer(
        key="humidity",
        label="Humidity",
        heading="Relative humidity map",
        unit="relative humidity (%)",
        colors=("#fef3c7", "#fde68a", "#a7f3d0", "#38bdf8", "#1e40af"),
        ticks=(0, 25, 50, 75, 100),
        lo=0.0,
        hi=100.0,
        note="Below ~60 % RH almost no indoor mould grows; 80 %+ is the activation threshold in FGPI.",
    ),
    "lux": Layer(
        key="lux",
        label="Light",
        heading="Light map (log scale)",
        unit="lux (log10 scale)",
        colors=("#0f172a", "#1e3a8a", "#0369a1", "#65a30d", "#fde047"),
        ticks=(0.0, 1.0, 2.0, 3.0, 4.0),
        transform=_log_lux,
        inverse=_lux_label,
        hi=float(np.log10(1.0 + 5000.0)),
        log_axis=True,
        note="Light is growth-neutral in the FGPI model (f_L = 1.0); it is shown for context only.",
    ),
}

DEFAULT_LAYER = "fgpi"


def get_layer(key: str | None) -> Layer:
    """Resolve a layer key, falling back to FGPI for unknown input."""
    return LAYERS.get(str(key or "").lower(), LAYERS[DEFAULT_LAYER])


def layer_keys() -> list[str]:
    return list(LAYERS)


def transformed(values: np.ndarray | list[float], layer: Layer) -> np.ndarray:
    return np.asarray(layer.transform(np.asarray(values, dtype=float)), dtype=float)