# X-PAIR frozen validation comparison

These are adapters around the official `xpair==0.1.1` model, not a modified
X-PAIR architecture. They preserve official checkpoint loading and FP32
Ankh-large inference, retain unrounded scores, and evaluate known interacting
pairs without the optional pair-interaction gate. This is validation only.

## Tracked and External Assets

- Frozen cohort: `data/benchmarks/xpair_val512_20261030/` contains the manifest,
  label-free input snapshot and separate evaluation labels.
- The frozen manifest retains its original CRLF bytes via a path-specific
  `.gitattributes` rule; do not reformat it because its SHA-256 is pinned.
- Metrics and hash receipts: `results/xpair_val512_20261030/`.
- Interpretation and limitations: `docs/XPAIR_VALIDATION_PILOT_20261030.md`.
- Weights, official source, embeddings and raw prediction NPZs are external
  assets. They are not redistributed in Git; `.gitignore` excludes model and
  feature binaries. See the locations below.

Official resource identities:

| Asset | Source / identity |
| --- | --- |
| Source wheel | PyPI `xpair==0.1.1`; SHA-256 `1bb6d7230300a492422e8f739925bb9e5dfdfc44d13af91c097cdba3ff611f06` |
| Interface checkpoint | `srescalli/X-PAIR_models`, revision `74cf6648ecb0904639f7830ec022c2be131a6798`; SHA-256 `7ab260dc7917205367cb56ba8f706b36ffa502540c44a5395a1a51457a4c6bd1` |
| Multitask checkpoint | Packaged in the official wheel; SHA-256 `d8c50d03cd6107d2b411d7095a34243ec1b31e6b7cc1d590329c9328c1e8df72` |
| Ankh-large | `ElnaggarLab/ankh-large`, revision `74b371dbfa3ee0a05d32ae74df0c2e0b82d6b9a6`; weight SHA-256 `517b6e8b279dedcb477af240b35c46bd6eb3307723eb281e60d4b2c8a87b889b` |

The source mirror's TLS failure was not bypassed. The verified official wheel
was extracted instead. Source bytes and checkpoint identities are checked at
inference time. Third-party license terms continue to apply.

## Existing Deployment

Local asset directory:
`/home/husrcf/Code/ProtBind/pypropel_unibio/benchmark_pipeline_tmp/pairwise_baselines_20261022/assets/X-PAIR`.

Official source directory:
`/home/husrcf/Code/ProtBind/pypropel_unibio/benchmark_pipeline_tmp/pairwise_baselines_20261022/official/X-PAIR-0.1.1`.

Star result aggregation:
`/data1/husrcf/benchmark/pairwise_baselines_20261022/batches/xpair_val512_20261030/`.

Use an existing compatible Python/PyTorch installation. Do not install or
replace torch, torchvision or torchaudio. The recorded run used the existing
AIAA Python with torch 2.13.0+rocm10.0.0 on AMD GPU1. Three pure-Python support
wheels were unpacked under `XPAIR_ASSETS/runtime_vendor`; they were not
installed into the environment. The expected versions are pytorch-lightning
2.5.1, torchmetrics 1.7.1 and lightning-utilities 0.14.3.

From the repository root, set `PYTHON` to the existing interpreter and
`XPAIR_ASSETS` / `XPAIR_SOURCE` to the external directories above. Set
`XPAIR_BASE_TORCH` to the expected existing torch package directory if different
from the recorded AIAA location. These path overrides do not change model math.

The asset directory must contain the wheel, `interface_pioneer.ckpt`,
`ankh-large/` and `runtime_vendor/`; the source directory must be the unmodified
wheel extraction, including `xpair/default_model/multitask_xfair.ckpt`.

```bash
PYTHON=/home/husrcf/anaconda3/envs/AIAA/bin/python
export XPAIR_ASSETS=/absolute/path/to/external/assets/X-PAIR
export XPAIR_SOURCE=/absolute/path/to/official/X-PAIR-0.1.1
export XPAIR_BASE_TORCH=/absolute/path/to/existing/site-packages/torch

"$PYTHON" scripts/benchmarks/xpair/check_runtime.py \
  --device cuda:1 --output /new/output/runtime.json

"$PYTHON" scripts/benchmarks/xpair/run_pilot.py \
  --input data/benchmarks/xpair_val512_20261030/val512_inputs.json \
  --output /new/output/val512_run --device cuda:1

"$PYTHON" scripts/benchmarks/xpair/evaluate_pilot.py score \
  --inputs data/benchmarks/xpair_val512_20261030/val512_inputs.json \
  --labels data/benchmarks/xpair_val512_20261030/evaluation_labels.json \
  --labels-sha256 d2f7e670f84cae102143782478d00d91fd0ed88d39680b4bb258b31960a6302c \
  --run /new/output/val512_run --output /new/output/val512_metrics
```

Output run/scoring directories must be new. The inference cache is shared under
the run directory's parent and protected by an exclusive writer lock. The input
and label exporters are provided for provenance, but will only accept the pinned
manifest and refuse to overwrite snapshots. They are not a generic cohort picker.

## Ours on the Same Cohort

`compare_ours.py` evaluates the preselected RCSB 8M epoch-5 checkpoint with its
original configuration. Supply `--source` as the frozen training source tree,
not the current development HEAD. It reads the original full manifest for
normalization and leakage guards, but only runs the selected validation pairs.
It refuses a different checkpoint, input snapshot or train manifest. It does
not train or perform threshold selection.

HPC3 deployment and raw results:
`/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1/_hpc3_training/20261017_v1/diagnostics/xpair_val512_20261030/result_v2/`.

The original frozen source is the sibling training directory's `code/`; the
checkpoint is `runs/large_batch8m/best.pt`. Its hash is pinned in the adapter.
The completed comparison used HPC3 job 780495. Local NPZ predictions are
excluded from Git but retained with the tracked summary receipt.

```bash
"$PYTHON" scripts/benchmarks/xpair/summarize_comparison.py \
  --inputs data/benchmarks/xpair_val512_20261030/val512_inputs.json \
  --labels data/benchmarks/xpair_val512_20261030/evaluation_labels.json \
  --ours /absolute/path/to/ours_result_v2 \
  --xpair-metrics results/xpair_val512_20261030/metrics.json \
  --output /new/output/same_cohort_comparison.json
```

The final summarizer asserts exact labels and directional order, and uses the
same `p > 0.5` comparator for every model. AP and trapezoidal PR-AUC are separate.
Different models' training sets and information budgets remain different;
official pretraining-overlap audits are pending. This is a system-level
validation diagnostic, not an independent test or an architecture-only claim.
