# TMINTER3

Research implementation of partner-specific protein interface prediction.
Given a known interacting chain pair (A, B), predict two residue-level
interface vectors and, optionally, a residue-pair contact map.

This repository is a standalone extraction of the PDBTM pairwise experiments
from PPC. It is **not** a released pretrained general-purpose predictor.
It now also contains the full-RCSB partner-specific PPIS pipeline. The whole
repository is the pairwise project; it is not a subdirectory of the single-chain
M18 model. Canonical remote: <https://github.com/sunnylab-cb/TMINTER3>.

The PDBTM S0-S9 results below are validation results from one training seed;
the frozen test split has not been evaluated for that selection round. They
must not be mixed with the later RCSB validation cohort.

## Active experiment tracks

| Track | Split and features | Entry points |
| --- | --- | --- |
| PDBTM pairwise | Historical global pairwise-SI30 component split; ESM-C MLC, ESMFold 90D and ESM2 intra-chain contacts | [Original results](#current-results), [provenance](docs/PROVENANCE.md) |
| Full RCSB pairwise | Observed-coordinate sequences; detected MMseqs SI30/coverage-0.8 edges plus same-PDB joint components; ESM-C MLC and Protenix 90D, no ESM2 contact graph in the initial main experiment | [Active split contract](docs/CURRENT_SPLIT_CONTRACT.md), [HPC3 training](docs/HPC3_FORMAL_TRAINING_20261017.md) |
| Same-cohort baselines | Frozen RCSB val512; partner-specific 5.5-A labels; no test selection | [X-PAIR comparison](docs/XPAIR_VALIDATION_PILOT_20261030.md), [reproduction](scripts/benchmarks/xpair/README.md) |

No native inter-chain pose is supplied to these sequence/predicted-monomer
models. An MMseqs detected-edge audit is not an exhaustive global Biopython
identity certificate. Protein-ligand tasks and their external holdouts are not
the active PPIS training contract.

## Historical PDBTM Task and Inputs

- Labels: 5.5-A heavy-atom contact to the specified partner, not an
  all-partner union. Unknown residues are masked with -100.
- Split: frozen DeepTMInter-style global pairwise-SI30 component split,
  with chain-filtered manifests. Not an MMseqs clustering claim.
- Inputs: frozen ESM-C MLC embeddings (3456D), ESMFold-derived single-chain
  descriptors (90D), and an ESM2 intra-chain contact graph.
- Architecture: shared residue encoder, TCN/contact graph, optional symmetric
  low-rank outer-product adapter, residue heads and auxiliary contact head.
- S0-S9 train downstream parameters from scratch; pretrained input features
  remain fixed. They do not resume or load the single-chain M18 checkpoint.

## Layout

```text
src/ppcbind/          Model and dataset implementation, including dependencies
scripts/training/    Training and validation partner-use diagnostic
scripts/benchmarks/  External-system adapters and same-cohort evaluation
scripts/preparations/ Pair-label generation from existing coordinate features
scripts/slurm/       Configurable submission entry point
configs/experiments/ Original experiment configurations
tests/               Loss and adapter regression checks
results/             Lightweight final metrics and diagnostic summaries
data/benchmarks/     Frozen small benchmark inputs and separate label snapshots
docs/                Contracts, provenance, asset paths and experiment notes
archive/             Historical site-specific Slurm scripts (not portable)
```

## Environment

Use an existing platform-compatible PyTorch installation. **Do not overwrite
custom CUDA/ROCm torch, torchvision or torchaudio builds.** The HPC2 training
environment used PyTorch 2.9.1 + CUDA 12.8. Other runtimes require validation.
Additional Python packages: numpy, scipy, scikit-learn, PyYAML.
Python 3.10 or newer is required by the source syntax.

From the repository root, without installing anything:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=/absolute/path/to/existing/environment/bin/python
"$PYTHON" tests/test_followup_adapter.py
"$PYTHON" tests/test_scratch_losses.py
"$PYTHON" scripts/training/train_esm_pair.py --help
```

## Training

Prepare the external assets listed in [docs/ASSETS.md](docs/ASSETS.md).
Copy an existing YAML, change its data paths and use a **new** output directory:

```bash
"$PYTHON" scripts/training/train_esm_pair.py \
  --config configs/experiments/20261002_pairwise_followup/S5_site_only.yaml
```

Relative data/output paths resolve from the working directory. The trainer
refuses an existing output directory. Keep `evaluate_test: false` during
model selection. Freeze a model and validation threshold before test use.

Slurm example (adjust partition and resources to the local cluster):

```bash
sbatch --partition=i64m1tga800ue \
  --export=ALL,PROJECT_ROOT="$PWD",PYTHON_BIN="$PYTHON",CONFIG="$PWD/configs/experiments/20261002_pairwise_followup/S5_site_only.yaml" \
  scripts/slurm/train_pair.slurm
```

## Current results

New PF0/PF1 Pairformer-inspired trials are described in
[docs/PAIRFORMER_20261006.md](docs/PAIRFORMER_20261006.md). PF0 completed;
PF1 timed out during epoch20. The follow-up A/B/C trials are documented in
[docs/PAIRFORMER_READOUT_20261009.md](docs/PAIRFORMER_READOUT_20261009.md).

Validation checkpoint selection: directional-chain macro average precision.
The JSON field `pr_auc` is sklearn **average_precision_score**, not trapezoidal
integration of the PR curve. `Best F1` thresholds below are validation-selected.

| Variant | Macro AP | Pooled AP | AUROC | Best F1 @ gate |
|---|---:|---:|---:|---:|
| S0 independent | 0.4427 | 0.3598 | 0.7978 | 0.4393 @ 0.70 |
| S1 outer product | 0.4435 | 0.3767 | 0.8017 | 0.4472 @ 0.60 |
| S2 hard negatives | 0.4415 | 0.3834 | 0.8045 | 0.4448 @ 0.61 |
| S3 consistency | 0.4444 | 0.3774 | 0.8052 | 0.4480 @ 0.63 |
| S4 rank16 | 0.4449 | 0.3663 | 0.8052 | 0.4443 @ 0.68 |
| S5 no contact loss | **0.4494** | 0.3771 | 0.8035 | 0.4449 @ 0.70 |
| S6 no ratio loss | 0.4445 | 0.3812 | 0.8079 | 0.4508 @ 0.65 |
| S7 no smoothing | 0.4405 | 0.3677 | 0.8017 | 0.4385 @ 0.61 |
| S8 endpoint features | 0.4428 | 0.3738 | 0.8089 | 0.4433 @ 0.66 |
| S9 null pooling | 0.4426 | 0.3657 | 0.8051 | 0.4435 @ 0.67 |

Partner replacement lowers S1 macro AP from 0.4435 to 0.4349-0.4361 across
three fixed replacement seeds. S0 is unchanged. This is counterfactual
sensitivity, not evaluation on verified noninteracting protein pairs.

## Limitations and provenance

Identical monomer inputs can have different role-specific interface labels
in oligomers. A deterministic swap-equivariant model cannot fit conflicting
directional targets exactly. This limitation is not repaired by changing labels.
The coordinate-feature label generator does not itself fetch or reconstruct
biological assemblies; their provenance must be audited upstream.

These results are not directly comparable to single-chain all-partner-union
M18, published contact-matrix AP, or external datasets with different labels.
See [docs/PROVENANCE.md](docs/PROVENANCE.md). No new software/data license is
asserted here; upstream and third-party rights remain applicable.
