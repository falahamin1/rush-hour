# Rush Hour Benchmark: Analysis & Presentation Plan

Status as of writing: 5 methods (H-rep, V-rep, GNN, Flat MLP, CNN) x 3 difficulties
(10/12/15 moves) x 15 puzzles = 225 combos. 219/225 finished; the remaining 6
(all difficulty 15 — 5x puzzle_idx=12, 1x puzzle_idx=5/gnn) died to the 20h wall-time
limit and are being resubmitted via `jobs/retry_failed.sh`, resuming from their
saved checkpoints rather than restarting.

This is the roadmap for turning the full 225-combo result set into paper-ready
claims once that finishes. Nothing here is implemented yet except where noted —
the scripts already built (`plot_rh_results.py`, `generate_latex_tables.py`) cover
Phase 1 only.

---

## Phase 1 — Finalize the core comparison (done, just needs a rerun)

1. Confirm all 225 policies present: `ls policies/*.pth | wc -l`
2. `python plot_rh_results.py` → solve-rate + reward bar charts (3 difficulty panels each, mean ± std across puzzles, sample size labeled per panel)
3. `python generate_latex_tables.py` → main summary table (`rh_results_table.tex`) + per-puzzle longtable appendix (`rh_appendix_table.tex`)

**Deliverable:** 2 PDFs + 2 `.tex` tables, ready to drop into the paper.

**Known limitation to state explicitly wherever these numbers are cited:** each
(puzzle, method) combo is trained with a single random seed. The std bars mix two
different sources of variance — genuine puzzle-to-puzzle difficulty, and
training-run-to-training-run stochasticity (PPO can converge to different local
optima across seeds) — and currently can't be told apart. See Phase 5.

---

## Phase 2 — Statistical significance (not yet built)

The bar-chart error bars are descriptive, not inferential — they don't establish
that e.g. V-rep beating GNN at difficulty 12 (87% vs 53% solve) is a real effect
rather than puzzle-sampling noise. Plan for a `significance_tests.py` script that
reads `policies/*.pth` directly (same loading pattern as `plot_rh_results.py`) and
does, per difficulty:

- **Omnibus test — Friedman test** across all 5 methods, blocked by puzzle (paired
  design: same 15 puzzles seen by every method). Run separately for solve rate and
  for reward. Answers "is there any difference among methods at all" before
  looking at any specific pair.
- **Post-hoc pairwise tests**, only if the Friedman test is significant:
  - **McNemar's test** for solve rate (binary, paired by puzzle) — correct choice
    over a plain two-proportion z-test because it only uses the puzzles where two
    methods disagree, which is what paired binary data calls for.
  - **Wilcoxon signed-rank test** for reward (continuous, paired, non-parametric —
    safer than a paired t-test given n≈15 and the bimodal reward distribution
    already observed, values clustering near +10 or near −3 with little in
    between).
  - **Holm-Bonferroni correction** across the 10 pairwise comparisons (5 methods →
    C(5,2)=10 pairs) per difficulty, since testing every pair inflates false
    positives.
- **Effect sizes alongside p-values** — matched-pairs rank-biserial correlation
  for Wilcoxon, plain win/loss/tie counts for McNemar. Reviewers want to know how
  big the effect is, not just whether p<0.05.

**Deliverable:** a table of (method pair, difficulty, test, statistic, p-value,
effect size, significant-after-correction?) plus a short plain-English summary of
which comparisons survive correction — that summary is what actually goes in the
paper text, not a wall of p-values.

---

## Phase 3 — Efficiency framing (cheap — data already saved)

Every `policies/*.pth` already stores `train_seconds` and `model_state`, so this
needs no new training:

- Table: mean wall-clock training time per method per difficulty.
- Table: parameter count per method (from `model_state_dict` tensor shapes).
- Optional scatter: solve rate vs. param count, solve rate vs. train time — this
  preempts the "sure V-rep wins, but is it just bigger/slower" pushback.

---

## Phase 4 — Qualitative failure analysis (uses existing `replay()` tooling)

`comparison.py` already has a `replay()` function that renders a step-by-step PNG
rollout for one policy on one puzzle. Plan: pick 1-2 puzzles per difficulty where
methods sharply disagree (e.g. V-rep solves, GNN doesn't) and generate
side-by-side replay figures for the appendix — this shows *how* a method fails
(stuck oscillating vs. wrong first move vs. plain timeout), which is a much more
concrete story than a bar chart for a reader trying to understand what the
polytope representation is actually buying.

---

## Phase 5 — (Optional, more expensive) Seed-variance decomposition

If we want to preempt "is n=15 puzzles at 1 seed each actually enough" from a
reviewer, the fix is running 2-3 seeds per (puzzle, method) at a reduced puzzle
count (e.g. 5 puzzles x 3 seeds instead of 15 puzzles x 1 seed — similar total
compute). That would let us report a proper variance decomposition (how much
outcome variance is puzzle-driven vs. seed-driven) instead of conflating them.

This is a **design choice to make before running**, not after — flagging it now
so it's a deliberate decision later rather than a scramble once the paper's in
review. Not needed to proceed with Phases 1-4.

---

## Phase 6 — Paper integration

- **Main body:** Phase 1 summary table + solve-rate figure + the 1-2 headline
  significance claims from Phase 2, framed around the original baseline
  question — does per-object structure and permutation-invariance actually
  matter, or is spatial/convolutional structure (CNN) or raw capacity (Flat MLP)
  enough?
- **Appendix:** Phase 1 per-puzzle longtable, full Phase 2 significance table,
  Phase 3 efficiency tables, Phase 4 failure-mode figure.

---

## Sequencing note

Phases 1, 3, 4 need nothing beyond the current sweep design and can run the
moment all 225 policies exist. Phase 2 needs a new script but no new training.
Phase 5, if we decide to do it, needs a new training run and should be decided
before kicking that off rather than bolted on afterward.
