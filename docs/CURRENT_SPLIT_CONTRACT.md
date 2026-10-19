# Current PPIS split contract (2026-10-15, corrected scope)

## Active task

TMINTER3 predicts partner-specific protein-protein interface residues from a
pair of protein chains. This work is NOT protein-ligand site prediction and
the input universe has NOT been identified as a PDBbind-BioLiP joint set.

Use the complete >=50 aa observed-coordinate sequence universe derived from
the RCSB PPIS label manifests: 251267 exact unique sequences and 988743 chain
instances. The chain-instance count is not a count of distinct sequences or
training pairs. Preserve insertion codes, assembly context and partner-specific
labels; do not replace pair labels with all-partner union labels.

No external sets are protected: not COACH420, HOLO4K, old membrane val/test,
or the former external PPIS reference sets. Do not delete their audit outputs;
they belong to a separate historical comparison only.

The former fixed 185762 train / 65505 evaluation pools and the older externally
constrained 80/10/10 or 74/13/13 candidates are SUPERSEDED for the PPIS mainline.
The user subsequently approved a NEW no-exclusion 80/10/10 candidate and
training preparation on star. HPC2 job 13094571 completed this assignment:
201013 train, 25127 val, 25127 test unique sequences. Whole joint components
remain indivisible. No test-based model/threshold selection is permitted.

Active unique_split.tsv SHA256:
`8a8f2c34073e83b3aada046a707e6f02b3aab68afd93655ab4768aca7e1d1798`.
See `RCSB_TRAINING_LAUNCH_20261016.md` for paths, counts and execution status.

## Reusable inputs on HPC2

Root: `/hpc2hdd/home/shuang886/jhspoolers/PPC/experiments/tminter3_rcsb_20261009`.

- `manifests/observed_ge50_20261015/unique_sequences.tsv`
- `manifests/observed_ge50_20261015/chain_to_unique.tsv`
- `artifacts/mmseqs_ge50_20261015/clusters.tsv`
- `artifacts/mmseqs_ge50_20261015/hits.tsv`
- `artifacts/mmseqs_ge50_20261015/summary.json`

The self-search was performed BEFORE and independently of external-reference
exclusion. It can be reused without rerunning MMseqs. Its detected-edge graph
has 24016 components (8155 singletons), largest 17447 sequences, and 90309681
directed hit rows including self hits. Thresholds: SI>=0.30, query and target
coverage>=0.80, cov-mode 0, seq-id-mode 0, sensitivity 7.5, cluster-mode 1.

These local-alignment, coverage-qualified hits are NOT an exhaustive global
Biopython SI certificate. Existing Biopython calculations remain separate.

## Completed gates and remaining work

1. Completed: the >=50 aa no-exclusion grouping audit (job 13094063) gives
   10886 same-PDB joint components, largest 156306 (62.21%). See
   `PPIS_ONLY_SPLIT_STATUS_20261015.md`. Preserve all existing artifacts.
2. Completed: the largest component stays entirely in train. All 90309681
   existing directed hit rows were checked. Cross-split detected SI edges,
   exact sequences, PDB entries, assemblies and joint components are all zero.
3. Completed: HPC3 job 723548 materialized 1707865 undirected contacting pair
   contexts with two directional residue labels per record. Split counts are
   train 1420430, val 136802, test 150633. Both partners are >=50 aa. These are
   label-ready counts, not feature-ready or unique biological-interaction counts.
4. Audit complete feature coverage separately. >=50 aa eligibility does NOT
   mean feature-ready: 26906 sequences of length 50-99 lack predicted-structure
   complete features, and 85 >=100 sequences previously failed folding.
5. Feature preparation completed in 2345; fixed smoke 2348 passed (0:0).
   Formal job 2349 is submitted: scratch seed49, 15 epochs, 1150993 ready train
   pairs / 123863 ready val pairs, with automatic checkpoint continuation.
   Test evaluation is off. No smoke pair/length caps remain. The initial feature
   variant is ESMC + Protenix 90D, explicitly without an ESM2 contact graph.
   Do not claim exhaustive global Biopython SI separation from this audit.

Historical handoff and source clarification: `../.agents/EXTERNAL_HOLDOUT_AUDIT.md`.
