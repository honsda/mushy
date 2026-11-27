"""DHT22 temperature / relative-humidity driver.

Hardware: DHT22 data line on GPIO4 (config: sensors.dht22_pin), 3.3 V and GND,
with a 10 kOhm pull-up between data and 3.3 V.

Two driver stacks are supported so the same code works on Bookworm images:
  * adafruit-circuitpython-dht + Blinka (preferred)
  * legacy Adafruit_Python_DHT (fallback)

Set sensors.simulate: true in config.yaml or FUNGI_SIMULATE=1 for bench
testing without hardware. Simulation values are deterministic-ish and kept in
a plausible indoor mould-risk band; they are NOT calibration data.
"""

from __future__ import annotations

import math
import random
import time

from app import load_config, simulation_enabled


class SensorError(RuntimeError):
    """Raised when the DHT22 cannot produce a valid sample."""



def _pin() -> int:
    return int(load_config().get("sensors", {}).get("dht22_pin", 4))


def _simulated_reading() -> dict:
    """Smooth synthetic signal for development and CI (no hardware)."""
    now = time.time()
    # Slow oscillation simulates day/night drift; jitter keeps charts honest.
    temperature = 22.8 + 2.4 * math.sin(now / 2400.0) + random.uniform(-0.25, 0.25)
    humidity = 86.0 + 5.0 * math.sin(now / 3100.0) + random.uniform(-0.8, 0.8)
    # Rarely drop RH below the 80 % plateau to exercise moderate classes.
    if int(now // 300) % 11 == 0:
        humidity -= 15.0
    temperature = round(min(34.0, max(6.0, temperature)), 1)
    humidity = round(min(99.0, max(62.0, humidity)), 1)
    return {"temperature": temperature, "humidity": humidity}


def _read_circuitpython(pin: int) -> dict:
    try:
        import adafruit_dht  # type: ignore
        import board  # type: ignore
    except ImportError as exc:  # pragma: no cover - only on dev machines
        raise SensorError(
            "adafruit-circuitpython-dht/Blinka not installed. "
            "See README (sudo apt install libgpiod2; pip install -r requirements.txt)."
        ) from exc

    gpio = getattr(board, f"D{pin}")
    try:
        dht = adafruit_dht.DHT22(gpio, use_pulseio=False)
    except Exception as exc:  # construction can fail if GPIO is busy
        raise SensorError(f"DHT22 init failed: {exc}") from exc
    try:
        # The DHT22 fails periodically by design; retry a few times.
        for _ in range(3):
            try:
                temperature = dht.temperature
                humidity = dht.humidity
                if temperature is None or humidity is None:
                    raise SensorError("DHT22 returned no data")
                if not (-40.0 <= float(temperature) <= 80.0):
                    raise SensorError(f"DHT22 temperature out of range: {temperature}")
                if not (0.0 <= float(humidity) <= 100.0):
                    raise SensorError(f"DHT22 humidity out of range: {humidity}")
                return {
                    "temperature": round(float(temperature), 1),
                    "humidity": round(float(humidity), 1),
                }
            except (RuntimeError, SensorError):
                time.sleep(1.0)
        raise SensorError("DHT22 read failed after 3 attempts")
    finally:
        try:
            dht.exit()
        except Exception:
            pass


def _read_legacy(pin: int) -> dict:
    try:
        import Adafruit_DHT  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SensorError(
            "No DHT22 driver available. Install adafruit-circuitpython-dht "
            "or run with FUNGI_SIMULATE=1 for bench testing."
        ) from exc

    humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT22, pin)
    if temperature is None or humidity is None:
        raise SensorError("DHT22 read failed (legacy driver)")
    return {
        "temperature": round(float(temperature), 1),
        "humidity": round(float(humidity), 1),
    }


def read_temperature_humidity() -> dict:
    """Return {'temperature': float, 'humidity': float}.

    Raises SensorError on hardware or driver failure.
    """
    if simulation_enabled():
        return _simulated_reading()

    pin = _pin()
    try:
        return _read_circuitpython(pin)
    except SensorError as first_error:
        # Blinka's sysfs/libgpiod path is timing-sensitive on some kernels, and
        # on Bullseye it cannot read the DHT22 at all. The legacy C driver talks
        # to /dev/gpiomem directly, so try it before reporting the failure.
        try:
            return _read_legacy(pin)
        except SensorError:
            raise first_error
