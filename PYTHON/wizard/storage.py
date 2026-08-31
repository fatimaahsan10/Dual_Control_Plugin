"""
JSON persistence for ModelConfig -- every answer set the wizard's question
flow collects is saved here, not just held in Streamlit's in-memory
session state, so a model can be reloaded/reviewed/reused later (see the
wizard's design plan, "Persistent storage" requirement).

SAFETY NOTE: a model's `name` is free-form user-typed text (step 1 of the
question flow) that this module turns directly into a filename. Never
join it onto a path unsanitized -- `_safe_filename()` strips everything
but a conservative whitelist before it ever reaches the filesystem, which
is what stops a name like "../../etc/passwd" (or a Windows-reserved
device name like "CON") from doing anything other than saving a file
called "model.json" in the models directory.

Every function here is exception-safe: a corrupted or hand-edited JSON
file, a missing directory, or a bad name is always reported as a
StorageError with a plain-language message, never an uncaught
FileNotFoundError/JSONDecodeError/KeyError reaching the caller.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from wizard.schema import ModelConfig

MODELS_DIR = Path(__file__).parent / "models"
EXAMPLES_DIR = Path(__file__).parent / "examples"

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10)),
}


class StorageError(Exception):
    """Raised for any problem saving/loading/listing models -- always
    carries a plain-language `.message`, never a bare filesystem or JSON
    exception."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _safe_filename(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(name).strip())
    cleaned = cleaned.strip("_.")
    if not cleaned or cleaned.lower() in _WINDOWS_RESERVED:
        cleaned = "model"
    return cleaned[:100]  # keep filenames reasonable on every filesystem


def _path_for(name: str, models_dir: Path) -> Path:
    return models_dir / f"{_safe_filename(name)}.json"


def save_model(config: ModelConfig, models_dir: Path = MODELS_DIR) -> Path:
    """
    Writes `config` to `<models_dir>/<sanitized-name>.json`, creating the
    directory if needed, and returns the path written. Raises
    StorageError (never a bare OSError) if the name is unusable or the
    write fails.
    """
    if not config.name or not str(config.name).strip():
        raise StorageError("Can't save a model with no name -- give it a "
                            "name first.")
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        path = _path_for(config.name, models_dir)
        path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        return path
    except OSError as e:
        raise StorageError(f"Couldn't save the model file: {e}") from e


def load_model(name: str, models_dir: Path = MODELS_DIR) -> ModelConfig:
    """
    Loads `<models_dir>/<sanitized-name>.json` back into a ModelConfig.
    Raises StorageError with a plain-language message on any problem --
    missing file, invalid JSON, or a file that doesn't structurally match
    what ModelConfig.from_dict() expects (e.g. a required field was
    hand-edited away or renamed).
    """
    path = _path_for(name, models_dir)
    if not path.exists():
        raise StorageError(f"No saved model found named '{name}'.")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise StorageError(f"Couldn't read the model file: {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise StorageError(
            f"The saved model file for '{name}' isn't valid JSON ({e}) -- "
            f"it may have been hand-edited incorrectly.") from e
    if not isinstance(data, dict):
        raise StorageError(f"The saved model file for '{name}' doesn't "
                            f"contain a model (expected a JSON object).")
    try:
        return ModelConfig.from_dict(data)
    except (TypeError, ValueError, KeyError) as e:
        raise StorageError(
            f"The saved model file for '{name}' is missing or has "
            f"mismatched fields ({type(e).__name__}: {e}) -- it may be "
            f"from an incompatible version or have been hand-edited "
            f"incorrectly.") from e


def load_model_from_path(path: Path) -> ModelConfig:
    """Same loading/error-handling as load_model(), but for an explicit
    path -- used for the read-only examples/ directory, which isn't
    addressed by sanitized name."""
    path = Path(path)
    if not path.exists():
        raise StorageError(f"File not found: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except OSError as e:
        raise StorageError(f"Couldn't read '{path.name}': {e}") from e
    except json.JSONDecodeError as e:
        raise StorageError(f"'{path.name}' isn't valid JSON ({e}).") from e
    try:
        return ModelConfig.from_dict(data)
    except (TypeError, ValueError, KeyError) as e:
        raise StorageError(
            f"'{path.name}' has missing or mismatched fields "
            f"({type(e).__name__}: {e}).") from e


def list_models(models_dir: Path = MODELS_DIR) -> list[str]:
    """Returns saved model names (not filenames/paths), sorted, or an
    empty list if the directory doesn't exist yet -- never raises."""
    if not models_dir.exists():
        return []
    try:
        return sorted(p.stem for p in models_dir.glob("*.json"))
    except OSError:
        return []


def list_examples(examples_dir: Path = EXAMPLES_DIR) -> list[Path]:
    """Returns paths to the read-only example configs shipped with the
    wizard, or an empty list if none exist -- never raises."""
    if not examples_dir.exists():
        return []
    try:
        return sorted(examples_dir.glob("*.json"))
    except OSError:
        return []


def model_exists(name: str, models_dir: Path = MODELS_DIR) -> bool:
    return _path_for(name, models_dir).exists()
