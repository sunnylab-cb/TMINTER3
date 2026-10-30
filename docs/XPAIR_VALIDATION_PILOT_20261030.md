# X-PAIR validation pilot, 2026-10-30

Both official X-PAIR checkpoints completed genuine Ankh-large sequence inference
on the same frozen val512 cohort used by the BIPSPI+ pilot. This is the current
RCSB pairwise PPIS task, not single-chain M18 and not the old PDBTM validation set.
No test inference or test threshold selection was performed.

## Coverage and input

- 512 original biological-assembly pair contexts, 1,024 directional evaluations.
- 428 distinct ordered exact-sequence inputs; 532 distinct observed sequences.
- 159,658 unique input residues; longest sequence 2,268 aa.
- All 428 inputs completed under each checkpoint; no failed or skipped inputs.
- Evaluation: 289,902 residue observations, positive fraction 0.09603935.
- Exact input duplication was removed only for inference, then predictions were
  remapped to all original contexts. Different context masks were not unioned.
- Only sequences were supplied. No native pose, predicted structure, docking,
  evaluation labels or pair-level interaction gate entered inference.

Frozen manifest SHA-256:
`682a1815212000991f0ceb9ce2c784aa9085bebf8d0d41fa0e5436836eca4108`.
Label-free input snapshot SHA-256:
`ee61e1cb3a7f7fa1213c4f8289cc6fa2a027a260f3778c7509b81af34d7897e2`.
Separate evaluation-label snapshot SHA-256:
`d2f7e670f84cae102143782478d00d91fd0ed88d39680b4bb258b31960a6302c`.

Labels reuse the formal training pipeline's
`observed-polymer-heavy-atom-5p5-assembly-v2` contract. This run checked record
identity, contact bounds/uniqueness, contact-to-residue projections, counts and
observed masks. It did not independently reconstruct atom distances or assemblies.

## Results

| Checkpoint | Direction-macro AP | Pooled AP | Trapezoidal PR AUC | Pooled AUROC | F1@0.5 | MCC@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| interface_pioneer | 0.232806 | 0.170547 | 0.170499 | 0.680225 | 0.240922 | 0.155955 |
| multitask_xfair | 0.266787 | 0.226613 | 0.226535 | 0.729291 | 0.240302 | 0.176931 |

Primary classification uses the official TorchMetrics comparator `p > 0.5`.
Supplementary gates were selected on this validation pilot only, over
0.01--0.99 in steps of 0.01; the lowest gate wins ties:

| Checkpoint | Best pilot-val F1@gate | MCC at that F1 gate | Best pilot-val MCC@gate |
| --- | ---: | ---: | ---: |
| interface_pioneer | 0.250120@0.66 | 0.150798 | 0.156934@0.53 |
| multitask_xfair | 0.296124@0.84 | 0.208580 | 0.209874@0.87 |

Macro AP weights each original directional context equally. Pooled metrics
weight every observed residue observation equally. AP is sklearn average precision,
not trapezoidal area. Results remain full-precision float32; the official CLI's
three-decimal export and optional interaction gate were not applied.

Do not compare these pilot numbers directly with Ours on the full validation set
or partial BIPSPI+ coverage. The exact same-cohort comparison is now completed
below. Official pretraining overlap/SI audits remain outstanding; this is not
evidence from an independent sealed test.

## Same-cohort Ours comparison

HPC3 job 780495 completed successfully (exit 0, total allocation 3 min 2 sec).
Ours used the RCSB large-batch-8M checkpoint, epoch 5, selected previously by
full native validation macro AP, not by this pilot. No training, threshold search
or test evaluation was performed. The checkpoint and original feature
normalization were kept unchanged. All 512 contexts, 1,024 directions and
289,902 residue observations matched the frozen label export exactly.

| Model | Direction-macro AP | Pooled AP | Trapezoidal PR AUC | AUROC | F1@0.5 | MCC@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours RCSB 8M, epoch 5 | 0.301697 | 0.273880 | 0.273869 | 0.762259 | 0.289466 | 0.226445 |
| X-PAIR multitask_xfair | 0.266787 | 0.226613 | 0.226535 | 0.729291 | 0.240302 | 0.176931 |
| X-PAIR interface_pioneer | 0.232806 | 0.170547 | 0.170499 | 0.680225 | 0.240922 | 0.155955 |

The common classification comparator is `p > 0.5`, matching official X-PAIR.
Ours' training evaluator uses `>=`; four exact 0.5 ties account for the tiny
difference from its logged F1 0.289458 and MCC 0.226434. Ranking metrics agree
exactly with the original evaluator. No new gates were fitted for Ours.

Compared with multitask_xfair, absolute improvements are +0.034909 macro AP,
+0.047267 pooled AP (+20.86% relative), +0.032968 AUROC, +0.049164 F1 and
+0.049514 MCC at the common threshold. These are point estimates, not a
statistical significance claim. X-PAIR's separately pilot-tuned F1 is 0.296124
at 0.84, slightly above Ours' untuned F1@0.5 (difference 0.006658); it must not
be silently mixed into a common-threshold table.

This is a complete-system comparison: Ours uses ESM-C MLC and predicted
monomer structural descriptors; X-PAIR uses Ankh sequence embeddings only.
Neither is given a native inter-chain pose in this evaluation. Different
training data and information budgets prevent an architecture-only causal claim.

Ours checkpoint SHA-256:
`0dddf1302a02cb0330899d43db32a1bb4306302997ef65dc7470f9c5c06edbcd`.
Ours inference took 88.06 seconds; total evaluation/scoring took 88.97 seconds,
peak allocated VRAM 3,451,529,728 bytes. Feature generation is excluded, so
these timings should not be used for end-to-end efficiency comparisons.

Artifacts: `results/xpair_val512_20261030/same_cohort_comparison.json` and
`results/xpair_val512_20261030/ours_rcsb8m/{predictions.npz,summary.json}`.
The comparison receipt includes input, label, checkpoint, prediction and summary
hashes. Exact sequence/PDB overlap with Ours' training split was checked as zero;
that check alone is not a new all-pairs homology audit.

## Source and execution

- Source: official PyPI `xpair==0.1.1`, unmodified code, architecture and weights.
- Models: official `interface_pioneer.ckpt` and packaged `multitask_xfair.ckpt`.
- Features: official Ankh-large final hidden state, 1536D per residue, FP32;
  U/Z/O/B/J normalized to X as in the official embedding code.
- Tokenization was checked against exact normalized residue order and length for
  every sequence. Cache identity includes original sequence, tokenizer/config
  hashes, model weight hash, revision and dtype.
- Ankh revision: `74b371dbfa3ee0a05d32ae74df0c2e0b82d6b9a6`.
- Existing AIAA Python and torch 2.13.0+rocm10.0.0 were reused read-only.
  Torch/torchvision/torchaudio were not installed or changed. Three pure-Python
  support wheels were unpacked into a project-local runtime import directory.
- GPU1: AMD Radeon Pro W7900 Dual Slot. Both models and all embeddings completed
  in 129.03 seconds (plus initial hash checks/imports); peak allocated VRAM
  5,768,751,104 bytes. Three embeddings from the real smoke were reused.
- Independent gpt-6-luna/max review passed the inference and evaluation adapters
  after hardening provenance and output/label checks. Additional tests checked
  50 F1/MCC cases against sklearn and all 856 pair output files for exact coverage,
  residue length, dtype and finite scores.

## Locations

Local asset root:
`benchmark_pipeline_tmp/pairwise_baselines_20261022/assets/X-PAIR/`.

- Adapters: `export_val_inputs.py`, `run_pilot.py`, `evaluate_pilot.py`.
- Raw outputs: `runs/val512_20261030_v1/{interface_pioneer,multitask_xfair}/`.
- Metrics/per-direction tables: `runs/val512_20261030_v1/evaluation/`.
- Embeddings: `runs/embeddings_ankh_fp32/`.
- Small repo snapshots: `TMINTER3/results/xpair_val512_20261030/`.

Star aggregation:
`/data1/husrcf/benchmark/pairwise_baselines_20261022/batches/xpair_val512_20261030/`.
Raw predictions are under `predictions/`, with metrics under
`predictions/evaluation/`. Frozen input and separate label snapshots are also
preserved there. This is an aggregation copy; inference ran locally.
