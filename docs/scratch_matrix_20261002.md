# From-scratch pairwise experiments

User requested random initialization instead of resuming or loading M18.
The pending warm-start rerun 12902449 was cancelled and preserved separately.

## Frozen shared contract

- Existing chain-filtered SI30 split and partner-specific 5.5-A heavy-atom labels.
- Reuse fixed pretrained ESM-C, ESMFold descriptors and ESM2 contact features.
- All downstream model parameters start randomly; no checkpoint path,
  no resume, no frozen-base epochs. Zero-initialized residual output remains
  an architectural initialization, not inherited trained weights.
- Seed49, 20 epochs, base/pair LR 1e-4, batch64, token budget65536,
  pair-cell budget1,000,000, single-pair maximum6,000,000, workers8.
- Validation directional-chain macro AP selects checkpoints; test is not read.
- Atomic output-directory reservation; separate artifact directories.
- GPU telemetry every5s; 12-hour Slurm limit.

| Variant | Difference from S1 |
|---|---|
| S0_independent | No partner adapter in forward; no contact auxiliary loss |
| S1_opm | Rank8 outer product, 32 pair channels, contact loss weight0.2 |
| S2_hardneg | Negative term mixes 50% hardest and 50% uniform sampled negatives |
| S3_consistency | Auxiliary MSE between site score and top4 contact evidence, weight0.05 |
| S4_rank16 | Rank16 projection, same32 pair channels |

The negative budget is min(20 times positive contacts,65536,available negatives)
per pair. Uniform samples are drawn with replacement. Evaluation always uses
the canonical full-negative contact loss. Consistency is a weak surrogate,
not a probability-theoretic union; its calibration effect must be measured.
S0/S1 establish new scratch controls; old warm-start scores are historical,
not matched controls for the loss/representation ablations.

## Validation and gates

CPU unit tests passed: canonical loss arithmetic, hard-negative selection,
masked gradients and swap symmetry of consistency. Deployment py_compile
and shell syntax checks passed. Independent narrow review found no remaining
blockers after correcting scratch freezing defaults and masking positives.
Requested Spark model is unavailable in backend; review used gpt-6-sol xhigh.

GPU smoke job12902499 runs all five variants sequentially, one epoch each
on128 train/32 validation pairs. Full jobs use afterok:12902499 and
kill-on-invalid-dep, so no full run starts if any smoke variant fails.

Submitted full jobs: S0=12902506, S1=12902507, S2=12902508,
S3=12902509, S4=12902510. At submission the debug smoke was pending
MaxJobsPerAccount; the full jobs were dependency-gated.

## Launch failure and correction

The five-variant Python smoke 12902499 completed successfully in69 seconds.
All five original full jobs subsequently failed before training in3-12seconds
(exit6). Their shared wrapper passed PyTorch's bare GPU UUID to nvidia-smi,
which requires the GPU- prefix. No formal results were produced.

GPU probe12903267 verified the discrepancy on an allocated node. Both
telemetry wrappers now normalize the prefix while preserving GPU-/MIG- IDs.
Narrow independent review passed. Exact-wrapper launch check12903272 ran
the scratch S1 smoke successfully and produced GPU telemetry, unlike the
earlier Python-only smoke that missed the wrapper issue. Full reruns retain
unchanged model, initialization, data and hyperparameters.

Replacement jobs: S0=12903274, S1=12903275, S2=12903276,
S3=12903277, S4=12903278; dependency afterok:12903272.

## Input ambiguity audit

CPU Slurm job12902496 completed a fixed-seed sample of32 exact-sequence
homomer pairs from train and32 from validation. All32 in each split have
identical actual tensor features across sides. Of these,26 train pairs and
28 validation pairs have different side-specific residue labels.

This verifies an identical-input/conflicting-directional-target subset.
At deterministic evaluation a swap-equivariant model must output equal
site vectors for equal inputs. It therefore cannot exactly fit both target
vectors for these cases. Counts are sample findings, not full-population
estimates or a numerical bound on overall AP. No labels were merged, filtered
or changed to hide this limitation. Full tensor audit details are stored in
scratch_input_ambiguity_audit.json.

Later diagnostics should stratify homomers/heteromers and repeated contexts,
report complex-balanced metrics, and test partner sensitivity. Alternatives
for ambiguous assembly roles require a separately specified multi-hypothesis
or union-target protocol; simply increasing network size cannot resolve
identical inputs with contradictory directional targets.
