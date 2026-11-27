"""Light sensor driver: BH1750 over I2C (preferred) or MCP3008 over SPI.

Contract: read_lux() -> float, always. Returns -1.0 when the sensor or driver
is unavailable, so a light failure never blocks temperature/humidity logging.

Wiring:
  BH1750   VCC 3.3 V, GND, SDA GPIO2 (pin 3), SCL GPIO3 (pin 5), ADDR -> GND
  MCP3008  VDD/VREF 3.3 V, AGND/DGND GND, CLK GPIO11, DOUT GPIO9,
           DIN GPIO10, CS GPIO8 (SPI0 CE0), CH0 from photodiode op-amp
"""

from __future__ import annotations

import math
import random
import time

from app import load_config, simulation_enabled

BH1750_ADDR = 0x23  # ADDR pin low; 0x5C when ADDR is tied high


def _simulated_lux() -> float:
    now = time.time()
    # Weak indoor daylight cycle; dark at "night" but never exactly zero.
    base = 180.0 + 150.0 * math.sin(now / 5400.0)
    return round(max(0.0, base + random.uniform(-15.0, 15.0)), 1)


def _read_bh1750(bus_id: int) -> float:
    import smbus2  # type: ignore

    bus = smbus2.SMBus(bus_id)
    try:
        bus.write_byte(BH1750_ADDR, 0x01)  # power on
        bus.write_byte(BH1750_ADDR, 0x10)  # continuous, high-resolution mode
        time.sleep(0.18)  # integration time for H-resolution mode
        # A bare 2-byte I2C read: smbus2 has no SMBus.read(), and a block read
        # would prepend a command byte the BH1750 treats as a mode change.
        message = smbus2.i2c_msg.read(BH1750_ADDR, 2)
        bus.i2c_rdwr(message)
        data = list(message)
        raw = (data[0] << 8) | data[1]
        return round(raw / 1.2, 1)
    finally:
        bus.close()


def _read_mcp3008(bus: int, device: int, channel: int) -> float:
    import spidev  # type: ignore

    spi = spidev.SpiDev()
    spi.open(bus, device)
    try:
        spi.max_speed_hz = 1_350_000
        spi.mode = 0
        # MCP3008 single-ended read: start bit, SGL/DIFF=1, channel, padding.
        request = [1, (0x08 | channel) << 4, 0]
        reply = spi.xfer2(request)
        raw = ((reply[1] & 0x03) << 8) | reply[2]
        # Approximate photodiode+op-amp scaling; calibrate max_lux per rig.
        max_lux = float(load_config().get("sensors", {}).get("mcp3008_max_lux", 2000.0))
        return round(raw / 1023.0 * max_lux, 1)
    finally:
        spi.close()


def read_lux() -> float:
    """Return lux, or -1.0 if the light sensor is unavailable."""
    if simulation_enabled():
        return _simulated_lux()

    cfg = load_config().get("sensors", {})
    driver = str(cfg.get("light_driver", "bh1750")).lower()
    try:
        if driver == "bh1750":
            return _read_bh1750(int(cfg.get("i2c_bus", 1)))
        if driver == "mcp3008":
            return _read_mcp3008(
                int(cfg.get("spi_bus", 0)),
                int(cfg.get("spi_device", 0)),
                int(cfg.get("mcp3008_channel", 0)),
            )
        return -1.0
    except Exception:
        # Light is optional by contract: never let it break a sample cycle.
        return -1.0
