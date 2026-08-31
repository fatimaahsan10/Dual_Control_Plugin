import json

import pytest

from wizard.schema import ModelConfig, StateSpec, ActionSpec, MeasurementSpec, CostSpec
from wizard.storage import (
    save_model, load_model, load_model_from_path, list_models, model_exists,
    StorageError, _safe_filename,
)


def _sample_config(name="my_model") -> ModelConfig:
    return ModelConfig(
        name=name,
        states=[StateSpec(name="x1")],
        actions=[ActionSpec(name="u1", min=0.0, max=10.0)],
        dynamics={"x1": "u1 - x1"},
        measurement=[MeasurementSpec(name="y1", expression="x1")],
        cost=CostSpec(running="x1**2", terminal="0"),
    )


def test_save_then_load_round_trips(tmp_path):
    cfg = _sample_config()
    path = save_model(cfg, models_dir=tmp_path)
    assert path.exists()
    loaded = load_model("my_model", models_dir=tmp_path)
    assert loaded == cfg


def test_save_creates_directory_if_missing(tmp_path):
    models_dir = tmp_path / "nested" / "models"
    cfg = _sample_config()
    path = save_model(cfg, models_dir=models_dir)
    assert path.exists()


def test_save_overwrites_existing_file(tmp_path):
    cfg = _sample_config()
    save_model(cfg, models_dir=tmp_path)
    cfg.description = "updated"
    save_model(cfg, models_dir=tmp_path)
    loaded = load_model("my_model", models_dir=tmp_path)
    assert loaded.description == "updated"


def test_save_empty_name_raises_storage_error(tmp_path):
    cfg = _sample_config(name="")
    with pytest.raises(StorageError):
        save_model(cfg, models_dir=tmp_path)


def test_load_missing_model_raises_storage_error(tmp_path):
    with pytest.raises(StorageError, match="No saved model"):
        load_model("does_not_exist", models_dir=tmp_path)


def test_load_invalid_json_raises_storage_error(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(StorageError, match="valid JSON"):
        load_model("broken", models_dir=tmp_path)


def test_load_json_that_is_not_an_object_raises_storage_error(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "listfile.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(StorageError, match="doesn't contain a model"):
        load_model("listfile", models_dir=tmp_path)


def test_load_structurally_mismatched_json_raises_storage_error(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bad = {"name": "x", "states": [{"name": "x1", "unexpected_field": 1}]}
    (tmp_path / "x.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(StorageError, match="mismatched fields"):
        load_model("x", models_dir=tmp_path)


def test_list_models_empty_when_directory_missing(tmp_path):
    assert list_models(models_dir=tmp_path / "nope") == []


def test_list_models_returns_sorted_names(tmp_path):
    save_model(_sample_config("zeta"), models_dir=tmp_path)
    save_model(_sample_config("alpha"), models_dir=tmp_path)
    assert list_models(models_dir=tmp_path) == ["alpha", "zeta"]


def test_model_exists(tmp_path):
    assert not model_exists("my_model", models_dir=tmp_path)
    save_model(_sample_config(), models_dir=tmp_path)
    assert model_exists("my_model", models_dir=tmp_path)


def test_load_model_from_path_round_trip(tmp_path):
    cfg = _sample_config()
    path = save_model(cfg, models_dir=tmp_path)
    loaded = load_model_from_path(path)
    assert loaded == cfg


def test_load_model_from_path_missing_file(tmp_path):
    with pytest.raises(StorageError, match="not found"):
        load_model_from_path(tmp_path / "ghost.json")


# ----------------------------------------------------------------------
# SAFETY: model names are free-form user text used to build a filename --
# path traversal and filesystem-reserved names must never reach the
# actual filesystem path unsanitized.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("raw,forbidden_substring", [
    ("../../etc/passwd", ".."),
    ("..\\..\\windows\\system32\\evil", ".."),
    ("/etc/passwd", "/"),
    ("a/b/c", "/"),
    ("CON", None),  # windows-reserved device name
    ("nul", None),
])
def test_unsafe_names_are_sanitized(raw, forbidden_substring):
    safe = _safe_filename(raw)
    if forbidden_substring:
        assert forbidden_substring not in safe
    assert safe.lower() not in ("con", "prn", "aux", "nul")
    assert "/" not in safe and "\\" not in safe


def test_path_traversal_name_saves_inside_models_dir_not_outside(tmp_path):
    cfg = _sample_config(name="../../evil")
    path = save_model(cfg, models_dir=tmp_path)
    # Resolved save location must still be inside tmp_path.
    assert tmp_path.resolve() in path.resolve().parents or path.resolve().parent == tmp_path.resolve()
    assert path.parent.resolve() == tmp_path.resolve()
