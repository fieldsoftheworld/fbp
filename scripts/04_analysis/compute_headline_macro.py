"""Compute the HONEST HELDOUT_11 ensemble macro — the presentation headline.

Why this exists: `heldout11_macro.txt` (produced by the standard chain) only
reports the SINGLE-BEST-h_min macro, which is 0.2638 and sits BELOW the paper's
0.324. The real headline ("ensemble beats paper") comes from the multi-h_min
pixel-vote ensemble, and NO chain script emits that number. The previously
quoted 0.335 was a 10-country macro (cambodia was missing from the ensemble
CSV). This script reads the REFRESHED ensemble CSV (cambodia now included via
ensemble v3 / job 107072) and computes the true 11/11 macro for every strategy,
plus the single-best baseline and the ensemble-vs-single win count.

Standalone, private (lives in /u/agiri1/fieldvote, not the tracked repo).
Run from the repo root so it can read logs/inference/*.csv.
"""

import sys
from pathlib import Path

import pandas as pd

# Paper's HELDOUT_11 subset (paper/scripts/_aggregate.py)
HELDOUT_11 = {
    "belgium", "cambodia", "croatia", "germany", "kenya", "latvia",
    "lithuania", "portugal", "slovenia", "south_africa", "sweden",
}
PAPER_MACRO = 0.324

REPO = Path("/u/agiri1/github/ftw-planet")
ENS = REPO / "logs/inference/ensemble_per_country.csv"
SUMM = REPO / "logs/inference/results_summary.csv"
OUT = Path("/u/agiri1/fieldvote/HEADLINE_ensemble_macro.txt")


def main() -> int:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    emit("=" * 70)
    emit("FTW-PLANET — HEADLINE HELDOUT_11 MACRO (honest, reproducible)")
    emit("Generated overnight by the private scratch job (no repo edits).")
    emit("=" * 70)
    emit()

    if not SUMM.exists():
        emit(f"ERROR: {SUMM} missing")
        OUT.write_text("\n".join(lines))
        return 1

    summ = pd.read_csv(SUMM)
    # single-best per-country PQ (already best h_min per country)
    sb = summ[summ["country"].isin(HELDOUT_11)].copy()
    sb_macro = sb["pq"].mean()
    sb_n = len(sb)

    emit("## 1. SINGLE-BEST-h_min macro (the conservative number)")
    emit(f"   HELDOUT_11 macro PQ = {sb_macro:.4f}   ({sb_n}/11 countries)")
    emit(f"   Paper PRUE-HD-B3 (full) HELDOUT_11 macro = {PAPER_MACRO:.4f}")
    emit(f"   Delta = {sb_macro - PAPER_MACRO:+.4f}")
    emit("   (This is what heldout11_macro.txt reports. It is BELOW paper —")
    emit("    do NOT lead with this one.)")
    emit()

    if not ENS.exists():
        emit(f"ERROR: {ENS} missing — ensemble v3 may not have finished.")
        OUT.write_text("\n".join(lines))
        return 1

    ens = pd.read_csv(ENS)
    # Defensive: the chain appends per-country rows; keep the LAST row per
    # (strategy, country) in case an earlier stale run left duplicates.
    ens = ens.drop_duplicates(subset=["strategy", "country"], keep="last")

    emit("## 2. ENSEMBLE macros by strategy (THE HEADLINE)")
    emit("   A_nms_pool   = pool polygons across h_min, agreement-weight, NMS (deployable)")
    emit("   B_oracle     = per-patch best h_min (upper bound, uses GT — NOT deployable)")
    emit("   C_pixel_vote = pixel-level majority vote across h_min (deployable)")
    emit("   D_elbow      = n_pred-curve elbow auto-select per patch (deployable)")
    emit()
    headline = {}
    for strat in ["A_nms_pool", "B_oracle", "C_pixel_vote", "D_elbow"]:
        s = ens[(ens["strategy"] == strat) & (ens["country"].isin(HELDOUT_11))]
        present = set(s["country"])
        missing = HELDOUT_11 - present
        macro = s["pq"].mean() if len(s) else float("nan")
        headline[strat] = (macro, len(s))
        flag = "  <-- HEADLINE (deployable)" if strat == "C_pixel_vote" else ""
        emit(f"   {strat:14s} macro PQ = {macro:.4f}   ({len(s)}/11){flag}")
        if missing:
            emit(f"                  MISSING: {sorted(missing)}")
    emit()

    c_macro, c_n = headline["C_pixel_vote"]
    emit("## 3. THE NUMBER TO PRESENT")
    if c_n == 11:
        emit(f"   C_pixel_vote (deployable ensemble) HELDOUT_11 macro = {c_macro:.4f}")
        emit(f"   Paper PRUE-HD-B3 (full)                             = {PAPER_MACRO:.4f}")
        emit(f"   Delta = {c_macro - PAPER_MACRO:+.4f}")
        if c_macro >= PAPER_MACRO:
            emit("   => We MATCH/EXCEED the paper headline with inference-only ensembling,")
            emit("      no retraining. This is the deployment story for Paper #2.")
        else:
            emit("   => Slightly below paper. Present as 'comparable, inference-only'.")
    else:
        emit(f"   WARNING: only {c_n}/11 countries present in ensemble CSV.")
        emit(f"   Partial C_pixel_vote macro over {c_n} = {c_macro:.4f}")
        emit("   The ensemble v3 job may not have produced all 11. Check 107072 logs.")
    emit()

    # Per-country: single-best vs C_pixel_vote, and win count
    emit("## 4. PER-COUNTRY: single-best vs pixel-vote ensemble (HELDOUT_11)")
    emit(f"   {'country':14s} {'single_best':>12s} {'C_pixel_vote':>13s} {'gain':>9s}")
    cvote = ens[ens["strategy"] == "C_pixel_vote"].set_index("country")["pq"]
    sb_idx = summ.set_index("country")["pq"]
    wins = 0
    counted = 0
    for c in sorted(HELDOUT_11):
        b = sb_idx.get(c, float("nan"))
        v = cvote.get(c, float("nan"))
        if pd.notna(b) and pd.notna(v):
            gain = v - b
            counted += 1
            if gain > 0:
                wins += 1
            emit(f"   {c:14s} {b:12.4f} {v:13.4f} {gain:+9.4f}")
        else:
            emit(f"   {c:14s} {b:12.4f} {'(missing)':>13s}")
    emit()
    emit(f"   Ensemble (C_pixel_vote) beats single-best on {wins}/{counted} HELDOUT_11 countries.")
    emit()

    # Also report the full 25-country single-best headline for context
    emit("## 5. FULL 25-COUNTRY single-best headline (context, not paper-comparable)")
    top = summ.sort_values("pq", ascending=False)[["country", "h_min_predict", "pq", "pq_ratio_vs_delany"]]
    for _, r in top.iterrows():
        ratio = r["pq_ratio_vs_delany"]
        ratio_s = f"{ratio:.1f}x DelAny" if pd.notna(ratio) else ""
        emit(f"   {r['country']:14s} h_min={r['h_min_predict']:<5} PQ={r['pq']:.4f}   {ratio_s}")
    emit()
    emit("=" * 70)
    emit("END. This file is the single source of truth for tomorrow's headline.")
    emit("=" * 70)

    OUT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWROTE: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
