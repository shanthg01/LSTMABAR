import math

import pytest

from lstmabar.config import load_config
from lstmabar.physics.kb import (
    KBError,
    default_pedals_dir,
    load_kb,
    load_pedal,
    parse_value,
    pedal_from_dict,
    taper_fraction,
)


def _minimal(**over):
    d = {
        "id": "x",
        "name": "X",
        "family": "overdrive",
        "topology": "opamp_shunt_clip",
        "clipping": [{"stage": "clip", "device": "si", "location": "shunt"}],
        "components": {"R1": "10k", "C1": "100n"},
        "pots": {"gain": {"value": "100k", "taper": "A"}},
        "sources": ["https://example.org/x"],
    }
    d.update(over)
    return d


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("4.7k", 4700.0),
        ("0.047u", 4.7e-8),
        ("51p", 51e-12),
        ("2.2M", 2.2e6),
        ("100n", 1e-7),
        ("1µ", 1e-6),
        ("560", 560.0),
        (47, 47.0),
        ("1e3", 1000.0),
    ],
)
def test_parse_value(raw, expected):
    assert math.isclose(parse_value(raw), expected, rel_tol=1e-12)


@pytest.mark.parametrize("raw", ["", "abc", "4.7x", "-1k", "0", True, "4.7 kk"])
def test_parse_value_rejects(raw):
    with pytest.raises(ValueError):
        parse_value(raw)


def test_taper_endpoints_and_midpoints():
    for t in ("A", "B", "C"):
        assert taper_fraction(t, 0.0) == 0.0
        assert taper_fraction(t, 1.0) == 1.0
    assert taper_fraction("B", 0.5) == 0.5
    assert math.isclose(taper_fraction("A", 0.5), 0.1)
    assert math.isclose(taper_fraction("C", 0.5), 0.9)
    xs = [i / 20 for i in range(21)]
    for t in ("A", "B", "C"):
        ys = [taper_fraction(t, x) for x in xs]
        assert ys == sorted(ys)


def test_minimal_pedal_parses():
    p = pedal_from_dict(_minimal())
    assert p.c("R1") == 10e3
    assert p.pots["gain"].resistance(1.0) == 100e3
    assert p.default_knobs() == {"gain": 0.5}
    assert p.clipping[0].symmetric


@pytest.mark.parametrize(
    "over, match",
    [
        ({"family": "boost"}, "family"),
        ({"topology": "tube"}, "topology"),
        ({"id": "Bad-Id"}, "id"),
        ({"components": {"C1": "0.047"}}, "plausible"),  # missing 'u'
        ({"components": {"R1": "10q"}}, "R1"),
        ({"pots": {"gain": {"value": "100k", "taper": "Z"}}}, "taper"),
        ({"pots": {"gain": {"value": "100k", "taper": "A", "default": 2}}}, "default"),
        ({"sources": ["electrosmash"]}, "URL"),
        ({"sources": []}, "sources"),
        ({"clipping": []}, "clipping"),
        (
            {
                "clipping": [
                    {"stage": "a", "device": "si", "location": "shunt", "n_pos": 0, "n_neg": 0}
                ]
            },
            "n_pos",
        ),
        ({"extra": 1}, "unknown"),
    ],
)
def test_validator_rejects(over, match):
    with pytest.raises(KBError, match=match):
        pedal_from_dict(_minimal(**over))


def test_missing_key_named():
    d = _minimal()
    del d["pots"]
    with pytest.raises(KBError, match="pots"):
        pedal_from_dict(d)


def test_id_must_match_file_stem(tmp_path):
    path = tmp_path / "other.yaml"
    path.write_text("id: x\n")
    with pytest.raises(KBError, match="stem|missing"):
        load_pedal(path)


def test_repo_kb_is_valid():
    """Every pedal file in the repo loads and validates."""
    kb = load_kb()
    assert "ts808" in kb
    for pid, pedal in kb.items():
        assert pedal.path is not None and pedal.path.stem == pid
        assert pedal.sources


def test_config_points_at_pedals_dir():
    root = default_pedals_dir().parent
    cfg = load_config(root / "configs" / "base.yaml")
    assert (root / cfg.paths.pedals).resolve() == default_pedals_dir()
    assert cfg.physics.volts_per_full_scale == 1.0
