# RCSB pairwise PPIS split and training launch

Current update: smoke 2348 completed 0:0 in 75 seconds. Formal scratch
training is now submitted as star job **2349**, normal/16h, with per-step
checkpointing and automatic continuation. See RCSB_FORMAL_TRAINING_20261016.md.
The older failure/retry history below is preserved for provenance.

## Latest status: retry 2348 promoted to critical

At the user's explicit request, job 2348 was updated in place from urgent to
critical with `scontrol update JobId=2348 QOS=critical`. No duplicate job was
submitted. The one-hour limit and training configuration remain unchanged.
Star's actual QoS policy permits urgent to preempt only background; critical
can preempt normal/urgent/background using REQUEUE. At the update the job was
pending scheduler dispatch; the original urgent submission details below are
retained for provenance.

Job 2345 completed preparation but failed at the first training DataLoader
batch after 01:01:13 (exit 1:0). Allocated CUDA recognized an A100 80GB.
The adapter incorrectly requested raw-PS `vectors`; the real artifact schema
is `spatial_vector_features`, shape [L,8,3]. This was a field-name integration
bug, not an observed GPU OOM or wall-time cancellation. No checkpoint or
validation metric was produced; the original v1 run is preserved.

Preparation output is complete. Feature-file-ready train/val/test pairs are
1150993 / 123863 / 131829; exclusions are 269437 / 12939 / 18804. Train-only
normalization used 170465 exact unique sequences and 55003867 residues.
ESMC file presence is checked at preparation; full payload checks still occur
on load. These counts do not certify all ESMC payload contents.

The corrected adapter explicitly requires `spatial_vector_features` and keeps
shape/finiteness checks. Tests use the actual schema rather than the invented
alias, with missing-key/wrong-shape/NaN regressions. 26 related tests passed;
a gpt-6-luna/max narrow review found no blockers. A real star TRAIN pair passed
payload loading, label projections and collation ([1,229,90] and [1,100,90]
feature tensors, 20 contacts). This lightweight payload check is not a substitute
for the full constructor's train-UID checksum checks during training.

Retry job **2348**: partition gpu, QoS **urgent**, GPU1, CPU8, RAM96G, 1h.
It is PENDING (Resources) at submission. PREPARE_RCSB_ASSETS=0 reuses ready_v1;
all training hyperparameters remain unchanged. Only the output directory moves
to runs/rcsb_s5_noESM2contact_smoke_seed49_v2. Test evaluation remains off.
Log: /data1/wholeset/RCSBprot/training/ppis_ge50_v1/logs/train_2348.log.
The sections below retain the original launch details for provenance.

## Scope

Partner-specific PPIS only. No COACH420, HOLO4K, old membrane split or other
external-set exclusions. Sequence identity is computed on observed-coordinate
chain sequences; original residue identities, insertion codes and assembly
context remain traceable in the source tag/mapping files. The training target
is a 5.5 A heavy-atom contact with the specified partner, not all-partner union.

## Completed split

HPC2 job 13094571 assigned whole SI+same-PDB components with seed 42 to an
80/10/10 unique-sequence target. The largest component (156306 unique sequences)
is entirely in train. Nothing was excluded for overlap with external datasets.

| Unit | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Exact unique sequences, >=50 aa | 201013 | 25127 | 25127 |
| Chain instances | 817862 | 83821 | 87060 |
| PDB entries | 112333 | 16590 | 15808 |
| Biological assembly contexts | 146290 | 22212 | 20827 |
| Chain-instance residues | 211967240 | 24239364 | 23140711 |
| Unique sequences of length 50-99 | 22044 | 2545 | 2317 |

The assignment checked all 90309681 existing MMseqs directed hit rows.
Cross-split detected SI edges, exact sequences, PDBs, assemblies and joint
components were all zero. MMseqs settings: identity >=0.30, both coverages
>=0.80, cov-mode 0, seq-id-mode 0, sensitivity 7.5, cluster-mode 1. This is a
detected-edge, coverage-qualified certificate, NOT exhaustive global Biopython
pairwise SI separation. Existing global-SI jobs were not stopped.

HPC2 output:
`/hpc2hdd/home/shuang886/jhspoolers/PPC/experiments/tminter3_rcsb_20261009/artifacts/ppis_only_ge50_split_80_10_10_v1/`.

It contains unique_split.tsv, chain_split.tsv, components.tsv, pdb_split.tsv,
summary.json and COMPLETED. unique_split.tsv SHA256:
`8a8f2c34073e83b3aada046a707e6f02b3aab68afd93655ab4768aca7e1d1798`.

## Materialized pair labels

HPC3 job 723548 completed successfully in 6 min 9 s. Of 1903031 source
undirected contacting chain-pair contexts, 195166 have at least one partner
shorter than 50 aa and are excluded by the declared length rule.

| Label-ready unit | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Undirected pairs with two directional labels | 1420430 | 136802 | 150633 |
| Participating unique sequences | 195200 | 24291 | 24331 |
| Participating chain instances | 805658 | 82408 | 85850 |
| Participating PDBs | 105464 | 15763 | 15087 |
| Pair-weighted residues, both sides | 764873190 | 73827874 | 75086151 |
| Pair-weighted positive residues | 67846447 | 7283865 | 7682470 |
| Pair-weighted positive rate | 8.8703% | 9.8660% | 10.2315% |

The 1707865 label-ready pairs are not unique interaction types or independent
observations. Repeated sequences and assembly contexts are retained with their
partner-specific targets. 7445 split-assigned unique sequences do not occur
in a surviving pair because the >=50-aa rule applies to BOTH partners.
This is separate from feature availability.

HPC3 root:
`/data/user/shuang886/tminter3/artifacts/ppis_only_ge50_training_v1/`.

- split/: assignment summary, unique_split.tsv and COMPLETED.
- pairs/: pair_manifest.csv, 64 uncompressed JSONL label shards, summary.json,
  COMPLETED; ~3.9 GB including temporary manifest parts.
- Each CSV row records the JSONL shard and byte offset, both sequence UIDs,
  SHA1s, lengths, chain-instance IDs, PDB and assembly IDs.
- Label records retain contact index pairs and both positive projections;
  observed_sequence_only=true is explicitly checked by the loader.

Local summary copies:
`artifacts/ppis_ge50_training_launch_20261016/{split_summary,pair_summary}.json`.

## Star preparation and launch

Code: `/home/husrcf/Code/TMINTER3`.
Data: `/data1/wholeset/RCSBprot/training/ppis_ge50_v1`.

Job **2345**, rcsb_prepare_train: partition gpu, QoS normal, one GPU, 16 CPUs,
96 GB RAM, four-hour limit. At submission it is PENDING (Resources), while
jobs 2343 and 2344 occupy both GPUs. No existing job was stopped. All star
QoS require at least one GPU; a CPU-only preparation submission was rejected
with QOSMinGRES. Therefore preparation and smoke share one compliant allocation.

The separate I/O-only tmux session rcsb_ppis_sync_v1 finished copying the
pair/split artifacts before GPU allocation. It transferred 3475164664 bytes
of pair assets at about 107 MB/s, excluding temporary manifest parts, then
copied the completion marker and split files. Log: logs/sync_hpc3.log. The
Slurm job repeats resumable rsync before reading them; raw structures are not
moved. The completed tmux session exits normally.

On allocation, job 2345:

1. Completes artifact transfer from HPC3; completion marker is copied last.
2. Gates both endpoints on ESMC/Protenix file availability and writes excluded
   pairs and missing UIDs. ESMC payload provenance/shape/finiteness are checked
   again on load; preparation does not reread the entire ESMC corpus.
3. Fits residue-weighted raw-89D normalization using each ready TRAIN unique
   sequence once. Val/test raw features are checked but never enter fitting.
4. Runs a scratch-initialized one-epoch engineering smoke on 256 train and 64
   val pairs with both lengths <=512. No test loader, metric or threshold use.

This is NOT full-dataset training and must not be reported as such. The smoke
is for tensor alignment, CUDA access, gradients, memory and throughput.

Initial candidate: RCSB_S5_noESM2contact, ESMC layers 12/24/36 (3456D),
Protenix Mini raw spatial scalars normalized to 89D plus an availability flag,
TCN/global context, low-rank outer-product pair module. It supplies NO
experimental label coordinates as input. The RCSB ESM2 contact graph has not
been verified, so it is explicitly disabled; this is not exact S5 reproduction.

Outputs after successful preparation:

- ready_v1/ready_pair_manifest.csv
- ready_v1/excluded_pairs.csv and missing_unique.tsv
- ready_v1/normalization_stats.json and train_unique_ids.txt
- ready_v1/summary.json
- runs/rcsb_s5_noESM2contact_smoke_seed49_v1/
- logs/train_2345.log

## Remaining gates before formal training

- Report feature-ready counts independently of the full split. The prior
  feature audit found 26906 short (50-99 aa) sequences without complete PS and
  85 longer failures; this new stage must record actual pair-level exclusions.
- Confirm allocated CUDA access; login-shell NVML/CUDA visibility was not valid
  evidence of an available GPU.
- Measure smoke throughput before selecting a full-run wall time. The existing
  trainer checkpoints at epoch boundaries, so a full epoch must fit the budget.
- Freeze feature-ready membership, initialization, seed and validation selection
  before formal training. Test remains sealed until checkpoint selection.

## Code verification

33 focused local pytest tests passed (split 6, pair materialization 4,
train-only normalization 6, native adapter 17); bash -n passed for both Slurm
scripts. Narrow gpt-6-luna/max reviews found no remaining blockers after fixes
to held-out feature validation, integer contact indices, smoke/test isolation
and homomer-safe bounded worker-local feature caching. No environment packages
were installed or custom torch distributions modified.
