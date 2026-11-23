"""Parametric fallback must rank five species without any model file."""

from __future__ import annotations

import app.models.species as species


def _parametric_only(monkeypatch):
    monkeypatch.setattr(species, "_load_model", lambda: None)
    monkeypatch.setattr(species, "_sgd_proba", lambda *_a, **_k: None)


def test_returns_ranked_top_five(monkeypatch):
    _parametric_only(monkeypatch)
    predicted = species.predict_species(24.0, 90.0, 50.0)
    assert len(predicted) == 5
    scores = [p["suitability"] for p in predicted]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert all(p["confidence"] in {"low", "medium", "high"} for p in predicted)


def test_suitability_is_monotonic_in_humidity(monkeypatch):
    _parametric_only(monkeypatch)
    name = "Aspergillus versicolor"
    scores = []
    for rh in (50.0, 70.0, 85.0, 95.0):
        ranked = {p["species"]: p["suitability"] for p in species.predict_species(24.0, rh)}
        scores.append(ranked.get(name, 0.0))
    assert scores == sorted(scores), scores


def test_outside_envelope_reports_low_confidence(monkeypatch):
    _parametric_only(monkeypatch)
    ranked = species.predict_species(34.9, 61.0, 0.0)
    assert all(p["confidence"] == "low" for p in ranked)


def test_unknown_input_does_not_raise(monkeypatch):
    _parametric_only(monkeypatch)
    assert species.predict_species(24.0, 85.0, -1.0)
    assert species.predict_species("x", None)


def test_east_java_overlay_adds_regional_species(monkeypatch):
    """The overlay must be additive: A. flavus is not in the global table."""
    came_from_overlay = species._apply_overlay(
        [{"name": "Aspergillus versicolor", "T_min": 4, "T_opt": 25, "T_max": 37}],
        {
            "species": {
                "Aspergillus versicolor": {"T_opt": 27},
                "Aspergillus flavus": {
                    "T_min": 12,
                    "T_opt": 30,
                    "T_max": 42,
                    "RH_min": 75,
                    "RH_opt": 90,
                },
            }
        },
    )
    by_name = {c["name"]: c for c in came_from_overlay}
    assert by_name["Aspergillus versicolor"]["T_opt"] == 27
    assert by_name["Aspergillus versicolor"]["T_min"] == 4  # merged, not replaced
    assert by_name["Aspergillus flavus"]["T_opt"] == 30
