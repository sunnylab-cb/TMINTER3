# SI30 split isolation is not representative downsampling

## Code-level finding

The active contract uses MMseqs detected SI>=0.30, both-sided coverage>=0.80 edges plus same-PDB grouping to assign indivisible components to train/val/test. It preserves all eligible exact unique sequences rather than selecting one sequence per SI30 component. See `CURRENT_SPLIT_CONTRACT.md`. This detected-edge separation is not an exhaustive global Biopython SI certificate.

Exact sequence deduplication is used to compute/store ESMC embeddings and predicted-monomer features once per UID. It is not a deduplication of training examples.

`scripts/preparations/materialize_rcsb_ppis_pairs.py` constructs pair IDs from PDB and chain-instance identifiers (including assembly context). It rejects duplicate/reversed instances, but deliberately preserves different chain-instance pairs with identical endpoint sequences. It preserves partner-specific labels rather than all-partner union labels.

`src/ppcbind/data/rcsb_pair_dataset.py` loads every selected manifest row. `_load_side` returns shared UID-based monomer inputs; the pair/assembly identities are not model inputs. `PairTokenBatchSampler` in `scripts/training/train_esm_pair.py` packs every instance once per epoch; it does not balance SI components or exact-sequence-pair groups.

Thus SI30 isolation, exact feature deduplication, and sequence-group-balanced training are three separate operations. The first two are present; representative downsampling and group balancing are not. If an SI30-nonredundant training set was intended, the current implementation does not meet that interpretation.

## Validation label audit

Final reviewed script: `scripts/analysis/audit_repeated_pair_labels.py`.
Tests: `tests/test_repeated_pair_labels.py`, two passing tests.
HPC3 final job 736359 completed with exit0 in 34 seconds. Earlier job736357 produced provisional statistics without the extra contact-projection validation; do not use it as the final provenance.

The final audit validates every selected JSONL record's pair/PDB/assembly/split/UID fields, label contract, fully observed-coordinate sequence length, unique contact pairs and index ranges, and exact projection of contacts onto both positive masks. No test JSONL labels or training examples are evaluated. Only validation rows enter the provenance hash.

Local artifacts: `artifacts/diagnostics/20261019/label_diversity/{summary.json,ordered_sequence_groups.csv}`.
Remote: `/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1/_hpc3_training/20261017_v1/diagnostics/frozen_20261018/val_label_diversity_20261019_v2/`.
Validation-row hash: `8fe36cb7435d29e909a3372167d15db05d20ae8a8358e16c7a6326e8457298f0`.

| Quantity | Count |
|---|---:|
| Validation pair instances | 123863 |
| Directional records | 247726 |
| Ordered exact (query UID, partner UID) groups | 37049 |
| Groups with more than one directional record | 16839 |
| Groups with more than one positive-label pattern | 12484 |
| Records belonging to variable-label groups | 216006 (87.20%) |
| Distinct ordered-input + positive-mask combinations | 82376 |
| Extra occurrences beyond those distinct combinations | 165350 |

74.14% of repeated ordered-input groups contain more than one mask. These counts describe validation, not training. The 87.20% figure is NOT an error rate, fraction of contradictory residues, or accuracy ceiling. Some masks differ only slightly; the audit counts any exact difference.

The earlier 22274 unique sequence pairs were UNORDERED. This audit is directional: heteromer sequence pairs contribute two ordered keys, identical-sequence pairs one. Hence 22274*2-7499=37049. Homomer left/right directional records correctly share the same sequence-input key.

## Concrete context dependence

UID `useq_70e147b3b07002dcef39f6a808fd7b33a616ad4339c3721f75ae6a14d220b158`, used for both endpoints:
- 144 aa; 3600 directional records across 13 PDBs.
- 64 positive-mask patterns; most frequent identical pattern occurs762 times.
- 100 positions are positive in at least one context, none positive in every context.
- In PDB7m2t assembly1, A-versus-C query A has one positive residue; J-versus-A query J has seven. These selected patterns have Jaccard0. They are selected for low overlap, not random representative cases.

A second identical-sequence group, UID `useq_46629fca15fb0bcf1af1253009d473c88ed1a77c6d751f3f118c2afdad25de75`, has114aa,1800 directional records in PDB6yf7, and45 label patterns. This demonstrates that repetition need not arise from many PDB entries; one multichain biological assembly suffices.

## Implications and safe next experiment

Both exact duplicate input-label weighting and genuine context-dependent outputs are present. SI30 cannot resolve either within a split: even SI100 clustering would keep all chain contexts unless materialization/sampling changes.

The current UID-based inputs lack biological-assembly pose or partner-instance identity. They cannot uniquely distinguish all these observed context-specific outcomes from sequence inputs alone. This is an information limitation, not proof of bad labels or a quantified irreducible-performance bound. It does not isolate the cause of the RCSB/PDBTM gap; that would require a controlled training comparison.

Do not arbitrarily take a representative from each connected component: the giant SI/PDB component contains diverse tasks, and removing contexts discards legitimate interactions. A conservative candidate is train-only group-balanced sampling (exact sequence pair first, context second), while preserving original validation/test and reporting secondary group-macro metrics. Exact input+label duplicate occurrences can be downweighted with full provenance retained. Differing masks should not silently be merged into union labels because that changes the partner-specific task.

No training, splits, thresholds, or labels were changed in this audit.
