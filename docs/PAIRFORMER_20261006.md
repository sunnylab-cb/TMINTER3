# Pairformer-inspired PDBTM trials

## Scope

PF0/PF1 use S5's site loss, ratio weight0.25, smoothing0.05 and zero contact
auxiliary weight. Frozen split and pretrained feature sources are unchanged.
Downstream parameters are random, seed49;20epochs; validation directional-chain
macro AP selection; test remains unread. This is not an AF3 reproduction.

## Architecture

Shared single-chain M18 encoder -> joint grid Z of size (LA+LB)^2x32.
Initialization: directed endpoint projections, signed clipped intrachain
sequence offsets, cross-chain unknown-offset token, and intrachain ESM2
contact scores with edge-availability indicator. No chain identity embedding
is added to Z; swapping input chains permutes the grid and predictions.

The existing ESMFold90D descriptors enter the single-chain encoder. These
cached files do not contain raw coordinates; no ESMFold distance matrix is
claimed in this first experiment. Missing ESM2 edges are unknown, not negative
contact labels. No experimental interchain coordinate/contact feature enters Z.

PF0: two blocks, each single-to-pair endpoint update, pair transition,
cross-chain pair row/column mean pooling, and residue residual update.
PF1: same design plus gated incoming/outgoing triangle multiplication before
each pair transition. Triangle hidden16, pair channels32, single256.
Final single-state readout is zero-initialized and added to base site logits.
No triangle attention, single attention, paired MSA or diffusion head.

Triangle aggregation is scaled by1/sqrt(joint length) and normalized. Pair
representations are directed; chain swapping is a token permutation, not a
requirement that each pair channel equal its transpose. Contact labels are
used solely in the existing site training targets; compatibility maps are
not advertised as supervised physical contact probabilities.

## Resource and fairness controls

Incoming/outgoing triangle products and pair transitions are chunked by128
rows; blocks use non-reentrant activation checkpointing. AMP is unchanged.
Batch cost now uses (LA+LB)^2, not LA*LB. Batch grid budget4M, single-pair cap25M.
The observed train/val maximum joint length4600 (21.16M cells) fits this cap.
Oversized pairs fail explicitly; there is no silent truncation or filtering.

PF0/PF1 share batching and all task hyperparameters, but triangle modules add
parameters and compute. A positive result is not a strict equal-parameter
causal attribution. S5 batching differs because its grid is rectangular;
PF0 is the matched batching/architecture control for PF1.

## Validation before deployment

- Existing loss and outer-product adapter tests pass.
- New CPU tests cover both triangle equations against einsum, chunking,
  checkpoint/no-checkpoint gradients, chain swapping, padding invariance,
  zero-init site residual, and per-sample graph isolation.
- GPU gate: one real-data smoke epoch for each variant, then one full
  forward/backward/optimizer step on the longest real training pair, no crop.
- Separate immutable remote snapshot; existing experiments remain untouched.
- Full jobs must depend on successful GPU gate,12h Slurm limit and telemetry.

Primary mechanism reference:
https://www.nature.com/articles/s41586-024-07487-w

Follow-ups only after evidence: more blocks, triangle attention, pair-biased
single attention, and separately audited predicted-coordinate distance inputs.

## Submission record

Submitted on 2026-10-06 as shuang886@hpc2, partition i64m1tga800ue:

| Job | Role | Time limit | Dependency |
| --- | --- | --- | --- |
| 12931841 | PF0/PF1 smoke and longest-pair stress | 1 hour | none |
| 12931842 | PF0 transition-only, scratch seed49 | 12 hours | afterok:12931841 |
| 12931843 | PF1 triangle updates, scratch seed49 | 12 hours | afterok:12931841 |

Both full runs use kill-on-invalid-dep. At submission verification the smoke
was pending for Priority and both full runs were pending for Dependency.
GPU peak memory and runtime are not yet validated; no new model results exist.
Narrow independent code review found no blocker for launching the smoke gate.

## Result check on 2026-10-09

Smoke completed in 3m52s. Longest-pair stress peaks: PF0 16.09 GiB / 1.68s;
PF1 50.78 GiB / 13.93s. These are isolated training steps, not epoch throughput.

PF0 completed all 20 epochs in 3h53m24s. PF1 hit the 12-hour time limit
during epoch20 step825; 19 epochs have complete validation. This was not
an OOM or low-utilization cancellation. Best checkpoints remain available.

| Variant | Best epoch | Val chain macro AP | Val pooled AP | Val AUROC | Val best F1@gate |
| --- | --- | --- | --- | --- | --- |
| S5 historical baseline | 8 | 0.449448 | 0.377113 | 0.803508 | 0.444850@0.70 |
| PF0 | 3 | 0.444356 | 0.385864 | 0.809106 | 0.443880@0.61 |
| PF1, incomplete run | 16 | 0.445497 | 0.371486 | 0.799366 | 0.443919@0.64 |

All rows select epochs using validation chain macro AP. PF0 values come
from final_metrics.json; PF1 values come from its saved epoch16 record.
PF1 epochs average 2233.4s (37.2min). No test metrics were read.
PF1 improves macro AP by 0.001141 over PF0 but trails S5 by 0.003951;
single-seed evidence does not establish a triangle-update improvement.
Local records: results/pairformer/PF0 and results/pairformer/PF1.
Future Slurm Python invocations explicitly use -u; PYTHONUNBUFFERED=1
was already enabled in the executed runs. No automatic rerun was submitted.
