# RCSB PPIS diagnostic review bundle

Snapshot date: 2026-10-19 (UTC+08:00).

Start with [the diagnostic handoff](../../docs/RCSB_PAIRWISE_DIAGNOSTIC_HANDOFF_20261019.md).

Then read the [objective/context/input audit](../../docs/RCSB_OBJECTIVE_CONTEXT_AUDIT_20261019.md).
The GitHub repository is private; unauthenticated 404 responses are expected.

## Included

- run_full, run_large_batch4m, run_large_batch8m: original config.json and completed-epoch metrics.jsonl copied from HPC3. These are running-experiment snapshots, not final results.
- ready_v1/summary.json: source and feature-ready coverage counts. Test coverage counts are metadata, not test evaluation results.
- ready_v1/normalization_stats.json: frozen train-only structural-feature normalization.
- ready_v1/train_unique_ids.txt.gz: all 170,465 feature-ready training sequence UIDs; these are identifiers, not amino-acid sequences.
- ready_v1/smoke_train_pair_manifest.csv: 256 training pair rows.
- ready_v1/smoke_val_pair_manifest.csv: 64 validation pair rows.
- objective_audit/: complete train/val dynamic-batch coefficient summaries, train mask summary, validation assembly-context and mask-effect summaries, a frozen epoch-3 FP32 input/swap/padding probe, and complete validation internal-gap summary. Source/config hashes are embedded in the JSON files.
- objective_audit/validation_gap_sources.csv.gz and validation_ordered_sequence_groups.csv.gz: per-source and per-group descriptive records behind the new summaries, not complete training labels.
- objective_audit/pdbtm_graph_run_config.json: original graph-control configuration used for the matched sampler/loss comparison.

The smoke manifests document the schema and mapping. They are NOT the complete training/validation cohorts, and the formal runs do not use their small-cohort caps.

The repository also includes validation-only diagnostic evidence:

~~~text
artifacts/diagnostics/20261019/rcsb_epoch3/
artifacts/diagnostics/20261019/pdbtm_epoch3/
artifacts/diagnostics/20261019/label_diversity/
artifacts/diagnostics/20261018/
~~~

Large diagnostic CSVs are published as deterministic gzip files; their JSON summaries are uncompressed. To reproduce the descriptive strata, decompress a directional_metrics.csv.gz alongside its summary.json and run scripts/analysis/summarize_validation_strata.py with an unused output path.

## Not Included

This is a lightweight reproducibility/evidence bundle, **not a complete runnable training dataset**. It does not include:

- the complete ready_pair_manifest.csv (566,803,195 bytes on HPC3);
- full per-residue label JSONL shards;
- amino-acid FASTA records or full split assignments;
- ESM-C tensors, predicted structures, structural-feature tensors or checkpoints;
- held-out test predictions or per-residue test labels.

All original data remain unchanged at:

~~~text
/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1/
~~~

Training metadata were copied from:

~~~text
training/ppis_ge50_v1/ready_v1/
_hpc3_training/20261017_v1/runs/{full,large_batch4m,large_batch8m}/
~~~

Config paths retain their original cluster locations for provenance and must be remapped before execution elsewhere. Do not infer complete feature coverage or a global exhaustive SI certificate from this bundle.
