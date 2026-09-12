"""Top-k confidence painting and pseudo-softmax rules for instance outputs."""

import numpy as np
import pytest
from fieldvote.cache_da import paint_topk
from fieldvote.da_softmax import RULES, pseudo_softmax


def _instances():
    masks = np.zeros((3, 10, 10), dtype=np.uint8)
    masks[0, 0:6, 0:6] = 1  # A, conf .9
    masks[1, 4:10, 4:10] = 1  # B, conf .6, overlaps A on [4:6, 4:6]
    masks[2, 0:6, 0:6] = 1  # C, duplicate of A, conf .3
    confs = np.array([0.9, 0.6, 0.3], dtype=np.float32)
    return masks, confs


def test_paint_topk_orders_by_confidence():
    masks, confs = _instances()
    ids, c = paint_topk(masks, confs, 3)
    assert ids[0, 0, 0] == 1 and c[0, 0, 0] == pytest.approx(0.9)
    assert ids[1, 0, 0] == 3 and c[1, 0, 0] == pytest.approx(
        0.3
    )  # duplicate sits in layer 2
    assert (
        ids[0, 5, 5] == 1 and ids[1, 5, 5] == 2 and ids[2, 5, 5] == 3
    )  # triple overlap
    assert ids[0, 9, 9] == 2 and ids[1, 9, 9] == 0
    assert (ids[0] > 0).sum() == (masks.any(axis=0)).sum()


def test_pseudo_softmax_rules_sum_to_one_and_mark_contest():
    masks, confs = _instances()
    _, c = paint_topk(masks, confs, 3)
    for rule in RULES:
        prob = pseudo_softmax(c, rule)
        assert prob.shape == (3, 10, 10)
        np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-6)
        seg = prob.argmax(axis=0)
        assert seg[9, 0] == 0  # background
        assert seg[5, 5] == 2  # contested -> boundary
    ratio = pseudo_softmax(c, "ratio")
    # weak duplicate (0.3 under 0.9) is a mild contest: pixel stays field
    assert ratio[2, 0, 0] == pytest.approx(0.3 / 0.9) and ratio.argmax(0)[0, 0] == 1
    # lone instance pixel: pure field
    assert ratio[1, 9, 9] == 1.0 and ratio[2, 9, 9] == 0.0
    # low absolute confidences must still count as present
    low = pseudo_softmax(c * 0.1, "ratio")
    assert low.argmax(0)[9, 9] == 1 and low.argmax(0)[9, 0] == 0
    hard = pseudo_softmax(c, "hard")
    assert hard[2, 0, 0] == 1.0  # duplicate counts as contest under the hard rule


def test_marker_supported_split_keeps_strong_seams_merges_weak():
    from fieldvote.variants import run_variant

    # A (.9) | B (.8) touch along column 5; C (.2) is a weak duplicate inside A.
    masks = np.zeros((3, 12, 12), dtype=np.uint8)
    masks[0, 1:11, 1:6] = 1
    masks[1, 1:11, 6:11] = 1
    masks[2, 3:8, 2:5] = 1
    confs = np.array([0.9, 0.8, 0.2], dtype=np.float32)
    ids_raw, _ = paint_topk(masks, confs, 1)
    from scipy.ndimage import binary_dilation

    dil = np.stack(
        [
            binary_dilation(m.astype(bool), structure=np.ones((3, 3), bool))
            for m in masks
        ]
    )
    _, c = paint_topk(dil, confs, 3)
    prob = pseudo_softmax(c, "ratio")
    seg = prob.argmax(0).astype(np.uint8)
    inst = run_variant("nbg-msup-t0.5", seg, prob, markers=ids_raw[0])
    # A|B seam has ratio .8/.9 (boundary-dominant) -> kept; C never wins a pixel
    assert inst.max() == 2
    assert inst[5, 2] != inst[5, 9] and inst[5, 2] > 0 and inst[5, 9] > 0

    # weak competitor: B at .3 vs A at .9 -> seam ratio .33 is field-dominant -> merge
    confs_weak = np.array([0.9, 0.3, 0.2], dtype=np.float32)
    ids_raw_w, _ = paint_topk(masks, confs_weak, 1)
    _, cw = paint_topk(dil, confs_weak, 3)
    prob_w = pseudo_softmax(cw, "ratio")
    seg_w = prob_w.argmax(0).astype(np.uint8)
    inst_w = run_variant("nbg-msup-t0.5", seg_w, prob_w, markers=ids_raw_w[0])
    assert inst_w.max() == 1
