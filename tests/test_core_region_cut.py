"""``+R``: keep or drop whole parcels by their mean field probability."""

import numpy as np

from fbp.core.postprocess import drop_low_field, foreground
from fbp.methods import parse


def test_drop_low_field_is_a_region_decision() -> None:
    inst = np.zeros((10, 10), np.int32)
    inst[:, :4] = 1  # mean pf 0.4
    inst[:, 6:] = 2  # mean pf 0.9
    pf = np.zeros((10, 10), np.float32)
    pf[:, :4] = 0.4
    pf[:, 6:] = 0.9
    out = drop_low_field(inst, pf, 0.5)
    assert set(np.unique(out)) == {0, 2}
    assert np.array_equal(drop_low_field(inst, pf, 0.3), inst)


def test_pf15_plus_r25_keeps_a_pale_field_that_argmax_splits() -> None:
    # a disc with field probability 0.4 everywhere (background 0.55): argmax says
    # background, the loose pixel foreground + region cut keeps it as one parcel
    y, x = np.mgrid[:64, :64]
    disc = np.hypot(y - 32, x - 32) < 24
    prob = np.zeros((3, 64, 64), np.float32)
    prob[0] = 1.0
    prob[1][disc] = 0.4
    prob[0][disc] = 0.55
    prob[2][disc] = 0.05
    seg = prob.argmax(0).astype(np.uint8)
    assert parse("nbg-pb-h0.01-t0.3")(seg, prob).max() == 0
    out = parse("pf15-pb-h0.01-t0.3+R25")(seg, prob)
    assert out.max() == 1 and (out > 0).sum() == disc.sum()
    assert parse("pf15-pb-h0.01-t0.3+R50")(seg, prob).max() == 0
    assert foreground(seg, prob, "pf15").sum() == disc.sum()


def test_fill_adopts_the_pale_disc_and_leaves_the_confident_block_alone() -> None:
    y, x = np.mgrid[:64, :96]
    disc = np.hypot(y - 32, x - 24) < 18  # pale: field 0.4 ...
    patch = np.hypot(y - 28, x - 20) < 4  # ... with one confident fragment inside
    block = (y > 8) & (y < 56) & (x > 56) & (x < 88)  # confident: field 0.9
    prob = np.zeros((3, 64, 96), np.float32)
    prob[0] = 1.0
    prob[1][disc], prob[0][disc], prob[2][disc] = 0.4, 0.55, 0.05
    prob[1][patch], prob[0][patch], prob[2][patch] = 0.9, 0.05, 0.05
    prob[1][block], prob[0][block], prob[2][block] = 0.9, 0.05, 0.05
    seg = prob.argmax(0).astype(np.uint8)
    base = parse("nbg-pb-h0.01-t0.3")(seg, prob)
    assert base.max() == 2 and (base > 0).sum() == block.sum() + patch.sum()
    m = parse("nbg-pb-h0.01-t0.3+R25+F15")
    assert m.id == "nbg-pb-h0.01-t0.3+R25+F15"
    out = m(seg, prob)
    assert out.max() == 2
    assert np.array_equal(out[block] > 0, base[block] > 0)  # primary untouched
    assert (out[disc] > 0).all() and len(np.unique(out[disc])) == 1  # one disc
