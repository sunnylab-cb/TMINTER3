# RCSB training trend update, 2026-10-19

Read-only remote inspection around 00:02 Asia/Taipei. No jobs or configurations changed.

## Live jobs

| Run | Job | Current training position | Recent pairs/s |
|---|---:|---|---:|
| 1M | 728963 | epoch 5, 20225/133609 batches (15.1%) | 42.1 |
| 4M | 729720 | epoch 5, 20425/34609 batches (59.0%) | 51.9 |
| 8M | 730850 | epoch 4, 11750/17056 batches (68.9%) | 55.6 |

All three confirmed RUNNING by squeue. Training progress excludes validation.

## Native RCSB validation curves

| Run | Epoch | Train loss | Val loss | Macro AP | Pooled AP | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 1M | 1 | .706395 | .762448 | .297654 | .259059 | .760642 |
| 1M | 2 | .665943 | .762999 | .297189 | .256361 | .758138 |
| 1M | 3 | .660934 | .759713 | .291519 | .259072 | .760480 |
| 1M | 4 | .659434 | .764479 | .294657 | .259145 | .760670 |
| 4M | 1 | .720758 | .751790 | .299815 | .261226 | .758818 |
| 4M | 2 | .671004 | .762640 | .296239 | .261073 | .760761 |
| 4M | 3 | .654833 | .754018 | .299935 | .267493 | .765236 |
| 4M | 4 | .646624 | .767255 | .300317 | .262292 | .764835 |
| 8M | 1 | .733432 | .749188 | .293569 | .248251 | .755437 |
| 8M | 2 | .686395 | .751020 | .303289 | .263112 | .763905 |
| 8M | 3 | .669222 | .750484 | .300977 | .266679 | .764965 |

Sources: HPC3 `/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1/_hpc3_training/20261017_v1/runs/{full,large_batch4m,large_batch8m}/metrics.jsonl` and corresponding logs.

Interpretation: 1M ranking is nearly flat. 4M train loss continues decreasing but macro AP barely moves; epoch4 pooled AP falls relative to epoch3. 8M pooled ranking improves through epoch3 but macro AP peaks at epoch2 so far. Larger batches improve throughput and show a modest ranking advantage at equal epoch3, not evidence of a solved generalization problem. Weighted validation losses across batch settings use different batch averaging and should not be used alone to rank settings. This is a single-seed observation, not statistical significance.

## Completed PDBTM graph control

HPC2 array 13179672_0/1 completed with exit 0, all 20 epochs. Both select epoch8 by native validation chain-macro AP; neither evaluated test.

| Control | Epoch8 macro AP | Epoch8 pooled AP | Epoch20 macro AP |
|---|---:|---:|---:|
| Graph | .450803 | .377272 | .444003 |
| No graph | .445102 | .366448 | .437117 |

Graph benefit at the same epoch8 is +.005701 macro AP and +.010824 pooled AP. This does not establish equivalence, but does not support graph removal as the sole explanation of the large raw native-dataset gap. These are PDBTM-native metrics and must not be directly subtracted from RCSB-native metrics to estimate dataset expansion benefit.

The previous shared-cohort epoch1/3 findings remain separate: RCSB ranks better on that fixed 100-PDB intersection while probability loss is worse. No new shared-cohort evaluation was performed in this update. Whole-RCSB plateau therefore remains a domain-specific diagnostic question, not disproved by shared-subset gains.

## Full native-validation stratification launched

User approved full stratified diagnosis. Frozen epoch3 RCSB 1M export: HPC3 job **735601**, acd_u, one GPU, eight CPUs, 128GB, four-hour limit. Frozen epoch3 PDBTM graph export: HPC2 job **13188443**, i64m1tga800u, one GPU, eight CPUs, 128GB, two-hour limit. Both initially PENDING (Priority). No training was stopped or modified.

Exporter: `scripts/training/export_validation_strata.py`. Uses each checkpoint's frozen config, val loader, strict weights; writes directional_metrics.csv and a completion summary only after exact left/right pair coverage is verified. No test evaluation or threshold tuning.

Remote outputs:
- HPC3 deployment `diagnostics/frozen_20261018/strata_rcsb_epoch3/`.
- HPC2 `/hpc2hdd/home/shuang886/jhspoolers/PPC/experiments/tminter3_s5_graph_control_20261018/strata_graph_epoch3/`.

Summarizer: `scripts/analysis/summarize_validation_strata.py`. Requires completed validation provenance, complete unique left/right records and finite metrics. Groups: positive prevalence (zero, <=5%, <=10%, <=20%, >20%); sequence length (<100, 100-299, 300-599, 600-999, >=1000); identical/nonidentical partner sequence; validation sequence-pair multiplicity (1, 2-5, 6-20, >20). Reports macro AP, prevalence, AP-minus-prevalence heuristic, CE, Brier, probability bias, and equally weighted unique-sequence-pair AP. Native-domain comparisons remain descriptive and are not isolated training-data-size effects.
