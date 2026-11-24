"""Blended inference config, light neutrality and the DB migration script."""

from __future__ import annotations

import sqlite3

import app.models.species as species

def _no_models(monkeypatch):
    monkeypatch.setattr(species, "_load_model", lambda: None)
    monkeypatch.setattr(species, "_sgd_proba", lambda *_a, **_k: None)

def test_blend_settings_prefer_the_nested_block():
    assert species._blend_settings({"blend": {"enabled": True, "sgd_weight": 0.25}}) == (
        True,
        0.25,
    )
    # Nested block wins over the legacy flat key.
    assert species._blend_settings(
        {"blend": {"sgd_weight": 0.2}, "blend_sgd_weight": 0.9}
    ) == (True, 0.2)
    # Flat key alone still works.
    assert species._blend_settings({"blend_sgd_weight": 0.4}) == (True, 0.4)
    # Adapter switched off, or the whole blend block disabled.
    assert species._blend_settings({"sgd": {"enabled": False}}) == (False, 0.3)
    assert species._blend_settings({"blend": {"enabled": False}}) == (False, 0.3)
    # Junk falls back to the default weight instead of raising.
    assert species._blend_settings({"blend": {"sgd_weight": "wide"}}) == (True, 0.3)

def test_blend_renormalises_and_keeps_sgd_only_species():
    ranked = [
        {"species": "A", "suitability": 0.8, "confidence": "high"},
        {"species": "B", "suitability": 0.4, "confidence": "high"},
    ]
    blended = species._blend(ranked, {"A": 0.2, "C": 0.6}, 0.5)
    by_name = {entry["species"]: entry["suitability"] for entry in blended}
    assert by_name["A"] == 0.5  # (1-0.5)*0.8 + 0.5*0.2
    assert by_name["B"] == 0.2  # (1-0.5)*0.4 + 0.5*0.0
    assert by_name["C"] == 0.3  # 0.5*0.6, adapter-only species kept
    assert [e["species"] for e in blended] == ["A", "C", "B"]

def _with_blend(monkeypatch, block: dict) -> None:
    """predict_species() reads a deep copy of config, so patch the loader."""
    from app import load_config as real_load_config

    def patched(path=None):
        cfg = real_load_config(path)
        cfg.setdefault("species", {})["blend"] = dict(block)
        return cfg

    monkeypatch.setattr(species, "load_config", patched)


def test_blend_disabled_leaves_the_posterior_untouched(monkeypatch):
    """With blending off, an SGD opinion must not leak into the ranking."""
    _no_models(monkeypatch)
    monkeypatch.setattr(
        species, "_sgd_proba", lambda *_a, **_k: {"Aspergillus niger": 0.99}
    )
    _with_blend(monkeypatch, {"enabled": False})
    predicted = species.predict_species(24.0, 90.0, 50.0)
    names = [p["species"] for p in predicted]
    assert "Aspergillus niger" in names  # still a plausible hypothesis...
    assert predicted[0]["species"] == "Aspergillus versicolor"  # ...but not top


def test_blend_enabled_lifts_the_adapter(monkeypatch):
    """A confident adapter is allowed to reorder the same reading."""
    _no_models(monkeypatch)
    monkeypatch.setattr(
        species, "_sgd_proba", lambda *_a, **_k: {"Aspergillus niger": 0.99}
    )
    _with_blend(monkeypatch, {"enabled": True, "sgd_weight": 0.3})
    predicted = species.predict_species(24.0, 90.0, 50.0)
    assert predicted[0]["species"] == "Aspergillus niger"
    assert predicted[0]["confidence"] in {"low", "medium", "high"}

def test_bright_light_does_not_zero_a_species(monkeypatch):
    """5000 lx used to trip a hard triangular cutoff; light is neutral."""
    _no_models(monkeypatch)
    dark = species.predict_species(25.0, 90.0, 0.0)
    bright = species.predict_species(25.0, 90.0, 5000.0)
    assert dark[0]["suitability"] > 0
    assert bright[0]["suitability"] > 0
    assert bright[0]["species"] == dark[0]["species"]

def test_curve_names_matches_the_loaded_table(monkeypatch):
    names = species.curve_names()
    assert "Aspergillus versicolor" in names
    assert len(names) == len(set(names))

def test_migrate_db_is_idempotent(tmp_path, monkeypatch):
    import scripts.migrate_db as migrate_db

    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_file))
    conn.executescript(
        """
        CREATE TABLE readings (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, x REAL, y REAL,
          node_id TEXT, temperature REAL, humidity REAL, lux REAL
        );
        CREATE TABLE fgpi (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          reading_id INTEGER REFERENCES readings(id),
          ts INTEGER, fgpi REAL, risk_class TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("FUNGI_DB", str(db_file))
    first = migrate_db.migrate()
    assert any("v2: create" in change for change in first["changes"])
    assert first["integrity"] == "ok"
    assert "labels" in first["tables_after"]
    assert first["backup"].endswith(".bak-" + first["backup"].rsplit(".bak-", 1)[1])

    second = migrate_db.migrate()
    assert second["changes"] == ["already current"]
    assert "backup" not in second

def test_model_path_resolves_from_the_repo_root(tmp_path, monkeypatch):
    """`docs/retraining.md` says `app/models/trained/x.joblib`; that must work.

    A relative path used to resolve against app/models/trained/ unless it
    happened to start with "app/", so any other spelling silently fell back to
    the parametric tier.
    """
    from app import BASE_DIR, MODEL_DIR

    from app import load_config as real_load_config

    def _patched(model_path: str):
        def loader(path=None):
            cfg = real_load_config(path)
            cfg.setdefault("species", {})["model_path"] = model_path
            return cfg

        monkeypatch.setattr(species, "load_config", loader)
        return species._model_path()

    repo_relative = tmp_path / "somewhere" / "model.joblib"
    repo_relative.parent.mkdir(parents=True)
    repo_relative.write_bytes(b"x")
    monkeypatch.setattr(species, "BASE_DIR", tmp_path)
    assert _patched("somewhere/model.joblib") == repo_relative

    # A bare filename still means app/models/trained/<name>.
    bare = MODEL_DIR / "species_rf.joblib"
    monkeypatch.setattr(species, "BASE_DIR", BASE_DIR)
    assert _patched("species_rf.joblib") == bare

    # An absolute path is taken as given.
    assert _patched(str(repo_relative)) == repo_relative
