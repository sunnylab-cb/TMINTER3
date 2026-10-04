# External assets (not included in Git)

Historical HPC2 workspace:
`/hpc2hdd/home/shuang886/jhspoolers/PPC`

Paths below are relative to that workspace. Preserve sequence hashes and
chain identifiers when relocating; never substitute a different split.

| Asset | Path |
|---|---|
| Pair manifest and labels | features/contact_labels_pairwise_5p5A_20261001/ |
| Chain split/masks | features/contact_labels_5p5A/splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708/pypropel_chain_filtered_no_len_limit/chain_manifest.csv |
| ESM-C embeddings | features/esmc_600m_2024_12_mlc_l12_l24_l36/by_chain/ |
| Predicted structure descriptors | features/esmfold_pred_struct_sequence_scalar_chain_filtered/pt/ |
| ESM2 graphs | features/esm2_mlc_chain_cache_20260711/by_chain/ |
| Base dataset index | features/esm2_mlc_chain_cache_20260711/compact_pdb/ |

Checkpoint locations on HPC2:
- S0-S4: `runs/pairwise_scratch_<variant>_seed49_20261002_hpc2/best.pt`
- S5-S9: `runs/pairwise_followup_<variant>_seed49_20261002_hpc2/best.pt`
- Current best validation macro AP: variant `S5_site_only`.

The repository contains model/loader code and the label-generation stage,
not an end-to-end FASTA-to-all-features extraction pipeline. Data, pretrained
PLM weights, ESMFold outputs and training checkpoints require separate access.

Dataset size: 16,103 train pairs / 4,030 validation pairs / 3,063 test pairs.
Validation metrics cover 8,060 directional chain instances, not 8,060 unique
proteins. Test counts originate from preparation; training selection does not
load test features or compute test metrics.

`results/followup/*_smoke/` and diagnostic paths containing `smoke` are launch
checks only, not formal benchmark results.
