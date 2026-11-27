# Sensor calibration

The forecast is only as good as the inputs. DHT22 and BH1750 are cheap parts
with real, correctable error. Do this once per sensor, then again every few
months (or after a move).

All corrections live in `config.yaml`:

```yaml
sensors:
  temp_offset: 0.0        # °C added to every reading
  humidity_offset: 0.0    # %RH added to every reading
  lux_scale: 1.0          # multiplier for the BH1750
```

`app/sensors/__init__.py:apply_calibration()` applies them in the sampler and
clamps the result to the physical range, so a bad offset cannot push humidity
to 103 %RH or lux negative.

## 1. Temperature — ice bath

Needs: crushed ice, water, a glass, a known-good thermometer (a kitchen probe
is fine).

1. Fill the glass with crushed ice and just enough water to fill the gaps.
   Stir for 2 minutes; the slurry settles at 0.0 °C at sea level.
2. Submerge the DHT22 body (not the connector) and wait 3 minutes for the
   housing to equilibrate.
3. Note `T_measured`. `temp_offset = 0.0 − T_measured`.
4. Sanity-check near room temperature against the reference thermometer:
   two points catch a non-linear sensor better than one.

A single-point ice bath is enough for the ±0.5 °C accuracy class the DHT22
claims; do not average it away.

## 2. Relative humidity — saturated salt

Needs: an airtight jar, table salt, distilled water, the DHT22.

1. Put a layer of salt in a small cup inside the jar, add water until it is
   damp sand — **not** a solution with standing water.
2. Seal the sensor in the jar, above the salt but not touching it.
3. Wait 6–8 hours (this is the slow one; the head-space must saturate).
4. Sodium chloride holds **75.3 %RH at 25 °C** (74.9 % at 20 °C, 75.7 % at
   30 °C). Read `RH_measured`, then `humidity_offset = 75.3 − RH_measured`.

Magnesium chloride (33 %RH) or potassium chloride (84 %RH) are alternatives if
the sensor is badly off; one salt at 75 %RH is enough for a sensor used in the
40–90 %RH band where FGPI's `f_RH` responds.

## 3. Light — dark box

The BH1750 is factory-calibrated and usually within a few percent; you are
checking for a window, a gap in the enclosure, or a saturated sensor.

1. Seal it in a light-tight box (a shoebox with a lid; leave it 5 minutes).
   Expect **0 lx**. Anything above ~2 lx means stray light — fix, do not
   scale.
2. Point it at a known lamp if you have a lux meter; otherwise trust the
   datasheet and set `lux_scale: 1.0`.
3. Direct sunlight saturates the sensor near 54612 lx (the `0xFFFF/1.2`
   ceiling). A truncated reading is recorded as-is; lux is growth-neutral in
   FGPI, so saturation does not distort the forecast.

## 4. Apply and verify

```bash
nano config.yaml            # set the three numbers
sudo systemctl restart fungi-server fungi-sampler
python scripts/sample_loop.py --once
curl -s http://localhost:5000/api/latest | python -m json.tool
```

Confirm the reading moved by the amount you intended, not more. The sampler
logs both raw and calibrated values at DEBUG level:

```bash
journalctl -u fungi-sampler -f
```

## 5. What calibration cannot fix

- **Placement.** A sensor on a warm wall reads the wall, not the room. The
  offsets are meant to correct the sensor, so fix placement first, then
  calibrate.
- **Response time.** DHT22 humidity lags ~10 s; a 30 s sample interval is fine,
  but do not calibrate while the value is still moving.
- **Drift.** A sensor that needed more than ±5 %RH of correction is probably
  degrading; `humidity_offset` will band-aid it for a while.
- **Sensor aging.** Re-run the salt jar yearly. If the offset changes by more
  than 2 %RH per year, replace the part.

## 6. What the dashboard shows afterwards

The cards, the maps and the species panel all use calibrated values — there is
one reading, corrected once, at the sampler. The `readings` table stores the
calibrated numbers with the raw values logged to the sampler journal, so a
retrospective correction is possible but never silent.
