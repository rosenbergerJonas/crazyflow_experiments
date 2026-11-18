"""Load the safe reach configuration and check that malformed configurations are rejected."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pytest

from tests.conftest import module_requires_env

module_requires_env("safe-reach")

from crazyflow_experiments.safe_reach.config import (  # noqa: E402
    Config,
    load_config,
    overlay_path,
    parse,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Inner:
    """Nested table of the test schema."""

    weight: float
    mode: Literal["a", "b"]


@dataclass(frozen=True)
class Outer:
    """Test schema covering every supported field type."""

    count: int
    flag: bool
    name: str
    vector: tuple[float, float, float]
    values: tuple[int, ...]
    inner: Inner


TABLE = {
    "count": 3,
    "flag": True,
    "name": "x",
    "vector": [0, 1.5, 2],
    "values": [1, 2],
    "inner": {"weight": 1, "mode": "a"},
}


def _with(key: str, value: Any) -> dict:
    """TABLE with one dotted key replaced, or removed if value is ...."""
    table = {k: dict(v) if isinstance(v, dict) else v for k, v in TABLE.items()}
    *parents, last = key.split(".")
    node = table
    for parent in parents:
        node = node[parent]
    if value is ...:
        del node[last]
    else:
        node[last] = value
    return table


def test_parse_converts_values():
    parsed = parse(Outer, TABLE)
    assert parsed.vector == (0.0, 1.5, 2.0)
    assert all(isinstance(v, float) for v in parsed.vector)
    assert parsed.values == (1, 2)
    assert isinstance(parsed.inner.weight, float)
    hash(parsed)  # Hashable, so sections can be static arguments of jitted functions


@pytest.mark.parametrize(
    ("table", "error", "match"),
    [
        (_with("extra", 1), ValueError, r"Unknown keys \['extra'\]"),
        (_with("inner.weigth", 1.0), ValueError, r"inner\.weigth"),
        (_with("count", ...), ValueError, r"Missing keys \['count'\]"),
        (_with("count", "3"), TypeError, "count must be a int"),
        (_with("count", 3.0), TypeError, "count must be a int"),
        (_with("inner.weight", True), TypeError, "inner.weight must be a float"),
        (_with("flag", 1), TypeError, "flag must be a bool"),
        (_with("inner.mode", "c"), ValueError, "inner.mode must be one of"),
        (_with("vector", [1.0, 2.0]), ValueError, "vector must have 3 entries"),
        (_with("values", [1, "2"]), TypeError, r"values\[1\] must be a int"),
        (_with("inner", 1.0), TypeError, "inner must be a table"),
    ],
)
def test_parse_rejects_malformed_tables(table: dict, error: type[Exception], match: str):
    with pytest.raises(error, match=match):
        parse(Outer, table)


def test_base_config_round_trips():
    config = load_config()
    assert Config.from_dict(config.to_dict()) == config
    assert load_config(overrides=config.flat()) == config
    hash(config)


def test_overlays_and_overrides_apply_in_order(tmp_path: Path):
    base = load_config()
    overlay = tmp_path / "variant.toml"
    overlay.write_text('[drone]\nmodel = "cf2x_L250"\n')
    from_overlay = load_config(overlay)
    assert from_overlay.drone.model == "cf2x_L250"
    assert from_overlay.drone.dynamics == base.drone.dynamics  # Keys the overlay omits stay
    overridden = load_config(overlay, {"drone.model": base.drone.model})
    assert overridden == base


def test_misspelled_override_is_rejected():
    with pytest.raises(ValueError, match=r"drone\.modle"):
        load_config(overrides={"drone.modle": "cf2x_L250"})


def test_missing_overlay_is_rejected():
    with pytest.raises(FileNotFoundError, match="no_such_variant"):
        overlay_path("no_such_variant")


def test_unknown_dynamics_and_drone_are_rejected():
    with pytest.raises(ValueError, match="drone.dynamics must be one of"):
        load_config(overrides={"drone.dynamics": "sys_id"})
    with pytest.raises((ValueError, KeyError)):
        load_config(overrides={"drone.model": "no_such_drone"})
