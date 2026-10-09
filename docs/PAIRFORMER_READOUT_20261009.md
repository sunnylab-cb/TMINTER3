# Pairformer selective readout trial

## Hypothesis and matrix

Earlier PF0/PF1 changed S5's explicit outer-product initializer and learned
partner pooling as well as adding a joint grid. The small PF1 gain over PF0
does not isolate these changes from S5. Test the following scratch seed49 runs:

| Variant | Partner aggregation | Cross-chain rank8 outer initializer | Triangles |
| --- | --- | --- | --- |
| A_selective | learned softmax | no | yes |
| B_outer_triangle | learned softmax | yes | yes |
| C_outer_no_triangle | learned softmax | yes | no |

A vs previous PF1 tests pooling (historical run has only 19 completed epochs).
A vs B isolates adding the outer initializer; B vs C isolates triangles.
Parameter counts differ; these are mechanism ablations, not equal-capacity tests.

Each block learns one pair-score projection, shared across both chain
directions. Scores are softmax-normalized along the partner axis in FP32;
the weighted sum replaces the previous mean. The scores are latent pooling
weights, not calibrated contact probabilities. No contact-label loss is added.

B/C add a shared-projection, symmetrized rank8 outer product only to AB/BA.
BA receives the transposed AB update, preserving chain-swap equivariance.
The outer projection is chunked and activation-checkpointed. Existing AA/BB
sequence offsets and ESM2 contact features remain unchanged.

## Frozen contract

Same frozen global SI30 split, partner-specific 5.5A labels, ESMC embeddings,
ESMFold90D and ESM2 intrachain graph; no experimental interchain input.
Same 20 epochs, seed49, scratch downstream initialization, losses, optimizer,
LR schedule, joint-grid batch budget4M and singleton cap25M in all three.
No chain crop/drop. Validation chain-macro AP selects checkpoints; test is
not read. The 24h Slurm limit changes available runtime, not epoch budget.

## Checks and deployment

CPU tests pass all old/new variants: zero-initialized residual, chain swap,
checkpoint/no-checkpoint gradients, graph isolation and sampler coverage.
GPU smoke runs all three short-manifest epochs and longest-pair training
steps. Full jobs require afterok on the shared smoke job and cancel if the
dependency fails. Each job requests one A800, 8 CPUs and160G host memory.
Python commands use explicit -u and GPU telemetry is logged every5 seconds.
Remote snapshot: experiments/tminter3_pairformer_readout_20261009.
Existing PF0/PF1 snapshot and outputs are not overwritten.

## Submitted jobs

Read-only gpt-6-sol xhigh review found no smoke blockers and additionally
checked finite gradients under CPU bfloat16 autocast. Requested Spark model
was unavailable. GPU validation is still pending.

| Job | Variant | Initial queue state |
| --- | --- | --- |
| 12954739 | shared smoke, 1h | Priority |
| 12954740 | A_selective, 24h | Dependency |
| 12954741 | B_outer_triangle, 24h | Dependency |
| 12954742 | C_outer_no_triangle, 24h | Dependency |

All full jobs depend on afterok:12954739 with kill-on-invalid-dep enabled.
Verified parameter counts: A3123718, B3127854, C3112622. Config equality
check confirms identical data and training fields except output_dir.
No new model performance results are available at submission.
