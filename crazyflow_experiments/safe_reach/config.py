"""Configuration of the safe reach experiment.

A run is described by `config.toml` in this folder, changed by overlay files and single overridden
values. Every table of the file is read into a frozen dataclass whose fields are exactly the keys of
the table, so a misspelled or missing key fails when the configuration is loaded instead of silently
falling back to a default. The values live in `config.toml` only, the dataclasses declare the keys,
their types and the checks on their values.

Overlays hold the keys a variant of the run changes, e.g. `configs/dual.toml`, and are merged over
the base file in the order they are given. Overrides are single values under dotted keys such as
`drone.model`, applied last. They are what the command line passes through fire, e.g.
`--drone.model cf2x_L250`.

The configuration is hashable, so its sections can be passed to jitted functions as static
arguments. TOML arrays become tuples for this reason.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args, get_origin, get_type_hints

import toml
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics

if TYPE_CHECKING:
    from collections.abc import Sequence

BASE_CONFIG = Path(__file__).parent / "config.toml"
OVERLAY_DIR = Path(__file__).parent / "configs"

T = TypeVar("T")


# region Sections
@dataclass(frozen=True)
class DroneConfig:
    """Drone model and the dynamics the simulator integrates."""

    model: str
    """Drone model, selects its physical parameters in crazyflow, e.g. `cf21B_500`."""
    dynamics: str
    """Dynamics model of the simulator, one of crazyflow's `Dynamics`, e.g. `so_rpy`."""

    def __post_init__(self) -> None:
        """Check that crazyflow knows the dynamics and has parameters of the drone for them."""
        names = [d.value for d in Dynamics]
        if self.dynamics not in names:
            raise ValueError(f"drone.dynamics must be one of {names}, got {self.dynamics!r}")
        load_params(Dynamics(self.dynamics), self.model)  # Raises for unknown drone models


@dataclass(frozen=True)
class Config:
    """Configuration of a run, one field per table of `config.toml`."""

    drone: DroneConfig

    @classmethod
    def from_dict(cls, table: Mapping[str, Any]) -> Config:
        """Read a configuration from nested tables, e.g. the dict saved with a checkpoint."""
        return parse(cls, table)

    def to_dict(self) -> dict[str, Any]:
        """Nested tables of the configuration, which `from_dict` reads back."""
        return dataclasses.asdict(self)

    def flat(self) -> dict[str, Any]:
        """Configuration under dotted keys, e.g. `drone.model`, as overrides and wandb use them."""
        return _flatten(self.to_dict())


# region Loading
def load_config(
    overlays: str | Path | Sequence[str | Path] = (), overrides: Mapping[str, Any] | None = None
) -> Config:
    """Load the configuration of a run.

    Args:
        overlays: Overlays merged over `config.toml` in the given order. Each is the name of a file in
            `configs/`, e.g. "dual", or a path to a TOML file.
        overrides: Single values under dotted keys, e.g. {"drone.model": "cf2x_L250"}, applied after
            the overlays.

    Returns:
        The checked configuration.
    """
    if isinstance(overlays, (str, Path)):
        overlays = (overlays,)
    table = toml.load(BASE_CONFIG)
    for overlay in overlays:
        table = _merge(table, toml.load(overlay_path(overlay)))
    table = _merge(table, _unflatten(overrides or {}))
    return Config.from_dict(table)


def overlay_path(overlay: str | Path) -> Path:
    """Locate an overlay given by its name in `configs/` or as a path to a TOML file."""
    path = Path(overlay)
    if path.suffix != ".toml":
        path = OVERLAY_DIR / f"{overlay}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"No overlay at {path}")
    return path


def parse(cls: type[T], table: Mapping[str, Any], key: str = "") -> T:
    """Read a table into the dataclass `cls`, checking every key and value against its fields.

    Args:
        cls: Dataclass with one field per key of the table. Fields that are dataclasses themselves
            read a nested table.
        table: The table to read.
        key: Dotted key of the table in the configuration, used in error messages.

    Returns:
        An instance of `cls`.
    """
    if not isinstance(table, Mapping):
        raise TypeError(f"{key or 'The configuration'} must be a table, got {table!r}")
    names = [f.name for f in dataclasses.fields(cls)]
    prefix = f"{key}." if key else ""
    if unknown := sorted(set(table) - set(names)):
        unknown = [prefix + k for k in unknown]
        raise ValueError(f"Unknown keys {unknown}, {key or 'the configuration'} has {names}")
    if missing := [prefix + name for name in names if name not in table]:
        raise ValueError(f"Missing keys {missing}")
    hints = get_type_hints(cls)
    return cls(**{name: _convert(table[name], hints[name], prefix + name) for name in names})


def _convert(value: Any, hint: Any, key: str) -> Any:
    """Check a value against the type of its field and bring it into the field's form.

    Supported types are nested dataclasses, `bool`, `int`, `float`, `str`, `Literal`, and tuples of
    fixed length (`tuple[float, float]`) or variable length (`tuple[float, ...]`). Integers are
    accepted for floats, since TOML writes 1.0 as 1 just as often. Booleans are never accepted as
    numbers.
    """
    if dataclasses.is_dataclass(hint):
        return parse(hint, value, key)
    origin, args = get_origin(hint), get_args(hint)
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{key} must be one of {args}, got {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{key} must be an array, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            args = (args[0],) * len(value)
        if len(value) != len(args):
            raise ValueError(f"{key} must have {len(args)} entries, got {len(value)}")
        return tuple(_convert(v, a, f"{key}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    accepted = {bool: (bool,), int: (int,), float: (int, float), str: (str,)}
    if hint not in accepted:
        raise TypeError(f"{key} has the unsupported field type {hint}")
    if not isinstance(value, accepted[hint]) or (isinstance(value, bool) and hint is not bool):
        raise TypeError(f"{key} must be a {hint.__name__}, got {value!r}")
    return float(value) if hint is float else value


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge `overlay` into `base`, recursing into tables both of them have."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten(table: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Turn nested tables into a single table with dotted keys."""
    flat = {}
    for key, value in table.items():
        if isinstance(value, Mapping):
            flat |= _flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _unflatten(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Turn dotted keys into nested tables, the inverse of `_flatten`."""
    table: dict[str, Any] = {}
    for dotted, value in flat.items():
        *parents, key = dotted.split(".")
        node = table
        for parent in parents:
            node = node.setdefault(parent, {})
        node[key] = value
    return table
