# PPIS-only split feasibility

Update 2026-10-16: the approved no-exclusion 80/10/10 assignment and pair
materialization have now completed. The historical feasibility analysis below
is retained; current execution details are in `RCSB_TRAINING_LAUNCH_20261016.md`
and the active authority is `CURRENT_SPLIT_CONTRACT.md`.

Scope correction: use the full RCSB partner-specific PPIS sequence universe,
not a PDBbind-BioLiP union. Protect no external datasets. The original version
of this document did not select or materialize split assignments.

## Completed MMseqs graph

HPC2 root:
`/hpc2hdd/home/shuang886/jhspoolers/PPC/experiments/tminter3_rcsb_20261009`.

Inputs: `manifests/observed_ge50_20261015/`.
Results: `artifacts/mmseqs_ge50_20261015/`.
Key files: `hits.tsv`, `clusters.tsv`, `summary.json`, `inputs.sha256`,
`mmseqs_version.txt`, and `GRAPH_AUDIT_COMPLETED`.

The self-search predates external-reference filtering and remains reusable.
251267 unique sequences, 988743 chain instances; 90309681 directed hits include
251267 self hits. The graph audit passed membership and detected-edge checks.
SI>=0.30, both coverages>=0.80, cov-mode 0, seq-id-mode 0, sensitivity 7.5,
cluster-mode 1. This is not an exhaustive global Biopython SI guarantee.

## New no-exclusion complex audit

Job 13094063 completed successfully (exit 0:0) in 10 seconds on HPC2 cpu2-15.
Existing code: `scripts/preparations/audit_complex_components.py`.
Result: `artifacts/mmseqs_ge50_20261015/ppis_only_complex_audit_v1.json`.
Explicit result fields: `external_protection=false`, `external_exclusions=0`.

| Constraint | Components | Largest unique component | Fraction |
| --- | ---: | ---: | ---: |
| SI30 / bidirectional cov80 detected edges only | 24016 | 17447 | 6.94% |
| SI plus same biological assembly | 10905 | 156247 | 62.18% |
| SI plus same PDB entry | 10886 | 156306 | 62.21% |

There are 144731 represented PDB entries and 189329 represented assemblies
among the >=50 aa chain records. These are not counts of complete surviving
training pairs; both partners still need the eligibility and feature gates.

The largest same-PDB component can be retained entirely in train. The other
94961 sequences (37.79%) are in 10885 components; the second-largest component
has only 534 sequences. Therefore a roughly 80/10/10 sequence-count target is
plausible without the former external-holdout exclusions. This is a feasibility
observation, not an exact achieved ratio or balanced pair/residue distribution.

## Original next steps (superseded by the 2026-10-16 update)

1. Choose a target allocation and generate candidate assignments at the joint
   component level; do not cut the giant component or reuse the old 185762 pool.
2. Build partner-specific pair manifests with both chains >=50 aa, preserving
   assembly IDs, original residue maps and pair labels. Count unique sequences,
   chain contexts, interacting pairs and residues separately.
3. Audit zero detected cross-split SI edges, exact-sequence overlap, reversed
   pair leakage and PDB/assembly overlap; inspect length/positive-rate balance.
4. Report features separately: the last audit has 26906 short sequences lacking
   complete PS and 85 longer sequences with missing PS. Do not silently remove
   them to obtain a desired split fraction or call them feature-ready.
5. Keep the ongoing global Biopython computation separate from this local,
   coverage-qualified MMseqs grouping. Its current progress was not rechecked
   during this focused scope-correction turn.

At the time of the original feasibility audit, no new sequence search,
folding, feature extraction or training had been launched. Only the short
read-only grouping audit had been submitted.
