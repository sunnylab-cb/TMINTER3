# Pairwise pilot audit, 2026-10-02

## Status

P1/P2/P4 completed six epochs. P3 job 12896022 was CANCELLED by UID
100001 (hpcadmin), after 09:25:58 of a 12:00:00 allocation, at
2026-10-01 23:54:08. Its batch step ended with signal 15. This was not
a Slurm time-limit termination. The user subsequently supplied the platform
notification: average GPU utilization below 20% for three continuous hours;
three warnings trigger automatic termination. The Reason field says
Dependency, which does not establish the cause of cancellation. Four epochs
completed; epoch five reached step 2850. Subsequent trials must address useful
GPU throughput before a full resubmission.

## Validation results

All values below use the checkpoint selected by validation directional-chain
macro AP. P3 is provisional, selected only among its four completed epochs.
Test outputs were not computed for these variants.

| Variant | Best epoch | Macro AP | Pooled AUPRC | AUROC | F1 at validation gate |
|---|---:|---:|---:|---:|---:|
| P1 site-only outer product | 4 | 0.43554 | 0.34757 | 0.79596 | 0.43760 @ 0.73 |
| P2 plus contact supervision | 4 | 0.44786 | 0.34677 | 0.79682 | 0.43756 @ 0.69 |
| P3 plus symmetric 2D convolution (partial) | 3 | 0.44360 | 0.35660 | 0.80170 | 0.43880 @ 0.72 |
| P4 plus Top-4 pooling | 4 | 0.44811 | 0.34215 | 0.79596 | 0.43497 @ 0.66 |

Prior validation macro AP: independent control 0.437805; cross-attention
0.440974. Single-seed differences are descriptive, not significance tests.

## Evidence and interpretation

1. Generalization gap, not simply too few epochs: P2 train site loss falls
   0.6520 -> 0.5189, while validation site loss rises 0.6822 -> 0.7044.
   Train contact loss falls 0.4085 -> 0.2152, while validation contact loss
   changes 0.4233 -> 0.4783. P4 has a similar trajectory. Frozen versus
   unfrozen epochs and dropout affect direct train/validation loss comparisons,
   but the later-epoch divergence remains visible.
2. Selection metrics differ: P2 epoch 2 has AUPRC 0.3566 and AUROC 0.8058,
   while macro-AP-selected epoch 4 has 0.3468 and 0.7968. Macro AP improvement
   alone does not prove reduced between-chain heterogeneity.
3. Manifest composition: train has 16,103 pairs / 2,933 PDBs; validation has
   4,030 pairs / 1,011 PDBs. Exact-sequence homomer pairs are 5,894/16,103
   (36.6%) in train and 3,230/4,030 (80.1%) in validation.
4. There are 5,221 distinct unordered sequence-pair keys in train and 1,098
   in validation. Of repeated-key groups, 1,734 train and 462 validation
   groups have different unordered positive-count pairs. This proves
   sequence-pair labels can differ across structural observations; it does
   not prove the full model inputs are identical, because chain and other
   context features must still be audited. Exact same-input conflicting
   targets would impose an irreducible ambiguity without assembly context.
5. Pair contact density is 0.0919% train and 0.0687% validation. Balanced
   positive/negative contact loss prevents simple class domination but does
   not concentrate the negative gradient on plausible interface pairs.
6. This is a small bilinear adapter, not a full AlphaFold pair stack:
   256D residue states are compressed to rank 8, expanded to a symmetrized
   8x8 product and 32 pair channels. P1/P2/P4 have no pair-grid interaction;
   P3 adds only local 3x3 sequence-index convolution. There is no relative
   inter-chain pose, triangle update, coevolution input or recycling.
7. P3 epochs cost about 7,130-7,237 seconds versus P2's 359-373 seconds
   after its first epoch. Per-pair Python execution and two convolution
   orientations are visible in the implementation, but profiling is needed
   before attributing the full roughly 20x slowdown to a specific kernel.
8. Pair labels are built from pre-existing all_atom_coords. The generator
   validates sequence hashes and within-PDB split consistency, but does not
   itself construct or verify biological assemblies. Assembly provenance
   remains an audit requirement, not a confirmed bug.

## Next diagnostics

- Stratify existing validation outputs by homomer/heteromer and repeated
  sequence pairs; use complex/component-aware uncertainty estimates.
- Test partner sensitivity by permuting partners while retaining query
  features, using validation only.
- Audit identical-input target conflicts and assembly provenance.
- Compare label-weighted and complex-balanced sampling and a frozen-base
  adapter control before increasing architectural complexity.
- Preserve the existing validation selection contract and sealed test.

Sources: copied Slurm stdout logs and final_metrics JSON files in this
directory; pair_manifest.csv; deployed payload esm_pair.py,
esm_pair_dataset.py and train_esm_pair.py.

## Performance diagnosis and repair

Slurm debug job 12902425 measured twelve real training batches on A40.
Excluding the first batch, total forward/backward time across eleven batches
was 34.410 s for the original P3 and 0.974 s for P3 with explicit contiguous
NCHW convolution inputs (35.3x). Median batch forward/backward was 2.474 s
versus 0.0685 s. The no-convolution comparison total was 0.834 s. These are
stage timings, not full training speedups: the profile freezes the base,
omits optimizer steps, and cycles three modes per batch, giving loader
prefetch extra time. End-to-end throughput and utilization need a full run.

The fix only calls contiguous() before the two symmetric Conv2d branches.
It preserves parameter names, shapes and mathematical operations. GPU test
12902440 compares against the original implementation with a nonzero site
residual head: FP32 and AMP output and input/parameter gradient max errors
were all exactly zero for the tested 165x645 shape; swap symmetry passed.
The earlier FP32-input synthetic microbenchmark (12902420) showed only a
modest gain and missed the pathological actual AMP pair-feature layout.

The optimized configuration changes only the output directory. The new
Slurm wrapper adds 5-second telemetry for the allocated GPU UUID, with
monitor cleanup on exit. No extra workload is used to inflate utilization.

Optimized six-epoch rerun submitted as Slurm job 12902449 on
i64m1tga800ue, after narrow independent code review. Output directory:
runs/pairwise_m18_opm_p3_conv_nchw_seed49_20261002_hpc2.
GPU UUID and telemetry process are checked before training starts.
