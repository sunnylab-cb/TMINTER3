# HPC3 ESMC extraction preparation

## Status

Formal extraction and structure transfer are complete. Array 728052 elements
0/1/2/3 all exited COMPLETED 0:0, with Slurm elapsed times 23m54s, 25m14s,
25m07s and 30m44s. Filename coverage audit found exactly 214,097 expected
outputs, zero missing/unexpected/misplaced files, and matching successful
extractor summaries covering 69,260,770 residues. Audit receipt:
`_esmc/completion_coverage_audit.json`. This is not an independent full tensor
reread; each tensor was checked for shape and finite values during extraction.
Structure-only transfer completed: 224,285 files, 118,426,410,343 bytes, with
zero missing PS endpoints in the transfer audit. Before training, verify
HPC3-local config paths and the dataset loader in a short training preflight.

Installation approved and completed on 2026-10-17. The fold environment was
cloned successfully; 50 pinned SDK/dependency packages were installed offline
with `--no-deps`, excluding unused torchtext. Star serves only the separate
structure-feature transfer; its ESMC feature bank is excluded.

Root: `/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1`

Scripts, manifests, weights and logs are under `_esmc/`. The exact model
snapshot was copied from the local machine, not the star feature bank.

## Verified inputs

The ready manifest was checked against the metadata transfer receipt; every
selected sequence was verified against its SHA-256 ID and both pair endpoint
lengths. Exact sequences do not cross train/validation/test splits.

| Split | Pairs | Unique sequences |
|---|---:|---:|
| Train | 1,150,993 | 170,465 |
| Validation | 123,863 | 21,662 |
| Test | 131,829 | 21,970 |
| Total | 1,406,685 | 214,097 |

Total residues: 69,260,770. Actual ready-sequence length range: 100-4091.
Although the parent split is named ge50, this extraction preserves the current
feature-ready subset; it does not silently expand to the entire parent split.
No labels enter feature extraction; copied test labels remain sealed for model
selection and normalization fitting.

Four length-interleaved shards contain 53,525 / 53,524 / 53,524 / 53,524 sequences.
Their residue totals are 17,316,738 / 17,313,646 / 17,314,699 / 17,315,687.
Inputs: `_esmc/manifests/ready_v1/{all_ready_unique,shard_00,shard_01,shard_02,shard_03,smoke}.tsv`.
Full source and shard digests are in the adjacent `summary.json`.

Ready manifest SHA-256:
`4d908b57ecba298697852d0ed808c9ed48989dfd77439db42b0f97f2f7d31fa8`.

## Frozen feature contract

- Model: `biohub/esmc-600m-2024-12`.
- Revision: `e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c`.
- Layers: 12, 24, 36 (1-based), final pretrained LayerNorm on each stream.
- Concatenation: 3456 dimensions, residue-aligned, no sequence clipping.
- Inference BF16; stored float32; external flash-attention disabled.
- Weight size: 2,300,275,866 bytes.
- Local and HPC3 weight SHA-256 both:
  `8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324`.
- Old HPC3 ESMC-600M snapshot `28aed...` and final-layer-only features are not
  substitutes for this contract.

The original extractor was copied with one cache-validation change: an existing
feature must also match the resolved model revision before it can be skipped.
Atomic output files permit restart with verified-output skipping. OOM currently
fails the shard explicitly; it is not silently converted into missing coverage.

## Environment approval required

`/data/user/shuang886/.conda/envs/fold/bin/python` has Python 3.12.11 and
Torch 2.7.1+cu128, but lacks the ESMC SDK, transformers and huggingface_hub.
The separate old `Folding/envs/esmc` environment lacks Torch as well.

Proposed environment: clone fold to
`/data/user/shuang886/.conda/envs/tminter3_esmc`, then prepare a dependency
installation dry-run. Preserve Torch/torchvision/torchaudio; do not modify fold.
User approved cloning and preparing the installation dry-run on 2026-10-17.
Reference local SDK versions:
esm 3.2.3, transformers 4.48.1, huggingface_hub 0.36.2.

### Dependency dry-run receipt

HPC3 could not connect to PyPI or the Tsinghua mirror. Compatible CPython 3.12,
glibc 2.28 wheels were downloaded on the local machine and copied to
`_esmc/wheelhouse/`; no feature bank was transferred.

The offline resolver completed successfully. Its machine-readable receipt is
`_esmc/install_dry_run.json`: 51 proposed package installations, **zero** changes
to torch, torchvision, torchaudio, triton or any nvidia-* runtime package.
Constraints are saved as `_esmc/hpc3_esmc_gpu_constraints.txt`.
The first dry-run used the source fold environment read-only while cloning was
in progress. The completed clone was subsequently checked again, producing
`_esmc/clone_install_dry_run.json`. Both resolved package/version maps are
identical, with zero protected GPU-runtime changes. Conda cloned 30 managed
packages and 34,658 additional files, exiting successfully.

Two installation caveats are recorded before changing anything:

1. The clone inherits fair-esm 2.0.0, which shares the `esm` namespace with the
   new SDK. Remove fair-esm only in the dedicated clone before installing esm.
   This clone is no longer intended for Protenix/fair-esm workloads.
2. esm declares torchtext as a dependency, but inspection of the 3.2.3 wheel
   finds no torchtext references in its Python source. The full 51-package plan
   includes torchtext 0.18.0; resolver success does not prove its binary ABI
   compatibility. An extraction-only install can omit it explicitly, record the
   resulting pip-check exception, and require ESMC import/forward smoke tests.

The old local zstd 1.5.7.3 wheel is yanked for a thread-safety bug and was not
copied to HPC3. Version 1.5.7.2 lacks a compatible binary wheel under the chosen
platform tags; the dry-run pins the available, non-yanked zstd 1.5.6.1 instead.
No model weights, hidden-state selection or normalization are changed by this.
This dry-run stage was followed by explicit user approval for installation
and real GPU extraction. Actual receipt: `_esmc/install_actual.json`; frozen
environment: `_esmc/environment_freeze.txt`. Torch remains 2.7.1+cu128,
torchvision 0.22.1+cu128 and torchaudio 2.7.1+cu128. Torch distribution RECORD
SHA-256 matches the original fold environment. ESMC class/tokenizer and
`huggingface_hub.load_torch_model` import checks passed.

`pip check` still reports unused inherited Protenix requirements/version pins
and the deliberately omitted torchtext dependency. This is a dedicated ESMC
environment, not a validated Protenix environment; no claim of a globally clean
pip check is made. The original fold environment is unchanged.

GPU smoke job 728047 was submitted to acd_u with array index 0 and 30-minute
wall time. It started on ACD1-67, reporting NVIDIA H100 80GB HBM3 (81559 MiB),
driver 610.43.02. It completed successfully (exit 0, 2m20s). All six outputs
passed an independent CPU-side check of exact sequence/hash, ID, L x 3456
shape, float32 dtype, finite nonzero residue vectors, layers, normalization,
revision and BF16 inference metadata. Lengths were 100, 256, 512, 1024, 2050
and 4091 (8,033 residues). Peak allocated VRAM was 1.80 GiB. The first batch
took 27.28 seconds including lazy initialization; subsequent batches took
0.08, 0.08 and 0.16 seconds. These mixed-length smoke timings are not a bulk ETA.
The model load itself took 55.42 seconds.

The ancillary read-only diagnostic Slurm step `728047.1` failed because the
main extraction process had already exited before its `/proc` lookup; the
main array task and batch step are both COMPLETED 0:0. This is not an extraction
failure.

Validated receipt: `_esmc/SMOKE_APPROVED.json`.

## Formal extraction launched

Array **728052**, elements 0-3, was submitted at 2026-10-17 05:50:50 HPC3
scheduler time. All four entered RUNNING at 05:50:54, with elements 0/1/2 on
ACD1-67 and element 3 on ACD1-32. Each requests one GPU, eight CPUs, 64 GB RAM
and 24 hours, on acd_u; the array throttle is four. No acd_ue allocation is used.

Logs: `_esmc/logs/esmc_728052_{0,1,2,3}.log`.
Outputs: `TMINTER3-ESMC/features/unique/<sha-prefix>/<unique-id>.pt`.
Features are generated on HPC3, not copied from star. The structure-only pull
continues separately. Training readiness still requires final PS/ESMC coverage
and content validation.

## Launch after environment acceptance

Launcher: `_esmc/scripts/hpc3_esmc_ready.slurm`.
Partition acd_u, four array tasks capped at four GPUs, one GPU each, eight CPUs,
64 GB host RAM, 24-hour wall time. Use absolute Python and `-u`.

First submit only array element 0 in `smoke` mode. Six sequences cover lengths
near 100, 256, 512, 1024, 2048 and 4091. Confirm shape, finite values, sequence
identity, provenance, GPU identity, memory and throughput before bulk launch.
Do not infer an ETA from model startup or an unverified GPU model.

Full mode requires a manually issued `_esmc/SMOKE_APPROVED.json` after actual
smoke inspection. Its `file_sha256` must bind the relative paths of the manifest
summary, smoke.tsv, extractor and launcher; `environment_sha256` binds the output bytes of
`python -m pip freeze --all`; `passed` must be true. The launcher also verifies
all shard hashes. An old approval cannot authorize changed scripts or inputs.

Full output: `TMINTER3-ESMC/features/unique/`. Keep structure-only transfer
completion distinct from training readiness. Audit complete ESMC/PS/label
coverage and tensor contents before starting HPC3 training.

Validation: four manifest/cache tests pass; launcher passes `bash -n`; remote
manifest preparation, model hash verification and six-sequence GPU smoke/QC
succeeded. Formal array 728052 is launched as documented above.
