"""The strategy-id grammar: parse/id round-trips and the registry."""

import pytest

from fbp.methods import REGISTRY, parse

SPECS = (
    "arg-none",
    "nbg-none",
    "otsu-none",
    "pf50-none",
    "arg-close1",
    "nbg-close2",
    "arg-ws2",
    "arg-ws2+A900",
    "nbg-pb-h0.01-t0.3+A900",
    "nbg-pb-h0-t0.3+A900",
    "nbg-pb-h0.01-t0",
    "nbg-sup-h2-t0.7",
    "nbg-sup-h2-t0.7+A900",
    "nbg-sup-h2-t0.7+l5",
    "nbg-sup-h2-t0.7+b1",
    "nbg-sup-h2-t0.7+f50",
    "nbg-sup-h2-t0.7+m50",
    "nbg-osup-h0.05-t0.7+A900",
    "nbg-ssup-h2-t0.6",
    "nbg-asup-h2-t0.5",
    "nbg-dsup-h2-t0.5",
    "arg-thr-te0.4-tb0.7",
    "nbg-ucm-k0.7",
    "arg-iterh-a200000",
    "nbg-swst-n80-m30-t0.3",
    "nbg-owt-l0.5-h0+A900",
    "nbg-gmm+A900",
)


@pytest.mark.parametrize("spec", SPECS)
def test_parse_id_round_trips(spec: str) -> None:
    m = parse(spec)
    assert m.id == spec
    assert parse(m.id) == m


def test_registry_covers_every_rule_keyword() -> None:
    for kw in (
        "none",
        "close",
        "ws",
        "pb",
        "sup",
        "osup",
        "ssup",
        "asup",
        "dsup",
        "thr",
        "ucm",
        "iterh",
        "swst",
        "owt",
        "gmm",
    ):
        assert kw in REGISTRY


def test_unknown_foreground_raises() -> None:
    with pytest.raises(ValueError, match="foreground"):
        parse("bogus-none")


def test_unknown_rule_raises() -> None:
    with pytest.raises(ValueError, match="rule"):
        parse("arg-bogus-h1-t0.5")


def test_unknown_modifier_raises() -> None:
    with pytest.raises(ValueError, match="unknown modifier"):
        parse("nbg-none+z5")


def test_gmm_needs_ctx() -> None:
    m = parse("nbg-gmm")
    assert m.needs_ctx == "gmm"
