# Full native-validation diagnosis, epoch 3

## Provenance

RCSB job 735601 completed successfully (21m07s). PDBTM job 13188443 failed before data loading because the original checkpoint uses paths relative to the PPC working directory. Resubmission 13199853 explicitly set that directory and completed successfully (1m03s). No checkpoint, feature, label, threshold or training setting was modified.

Exports and verified descriptive summaries are under `artifacts/diagnostics/20261019/{rcsb_epoch3,pdbtm_epoch3}/`: `directional_metrics.csv`, `summary.json`, `strata.json`. Both exports are complete, validation only, and have zero train/validation exact-sequence and PDB overlaps. Each pair has exactly one left and right record. This is not a new homology audit.

RCSB checkpoint SHA256: `e445f2a710ce676d10b2bca73659c8593963c61c78c5d40d83b9d80e098bb1f1`.
PDBTM graph-control checkpoint SHA256: `c7a1c2f496dc7c12a41a2afbf26bdfb6099cd79d9b1d40e0855786f177610375`.
The latter is the matched S5 graph-control retraining, not the original historical checkpoint.

## Main result

| Native validation | RCSB | PDBTM S5 graph control |
|---|---:|---:|
| Pair instances | 123863 | 4030 |
| Directional chain records | 247726 | 8060 |
| Unique unordered exact-sequence pairs | 22274 | 1098 |
| Directional-chain macro AP | .291519 | .432344 |
| Equal unique-sequence-pair AP | .398481 | .441163 |
| Macro positive fraction | .109488 | .155845 |
| Unweighted macro CE | .61898 | .54628 |
| Macro mean-probability minus prevalence | .32358 | .23155 |

Equal unique-pair AP means first averaging the directional APs of all structural instances of an unordered sequence pair, then averaging across unique pairs. It does NOT union or discard differing labels, and is not the original primary evaluation metric. Changing aggregation changes the estimand. The native-cohort gap shrinks from .140825 to .042682, but the cohorts and feature sources still differ; this is not a causal estimate of expansion damage or improvement.

## Duplicate-weight concentration

| Sequence-pair instance multiplicity | RCSB records | RCSB AP | PDBTM records | PDBTM AP |
|---|---:|---:|---:|---:|
| 1 | 27472 | .43170 | 1020 | .45663 |
| 2-5 | 30250 | .36554 | 2194 | .42803 |
| 6-20 | 43240 | .30714 | 3500 | .43674 |
| >20 | 146764 | .24542 | 1346 | .40954 |

Only 884 RCSB unique sequence pairs contribute the >20 group, representing 59.24% of all directional records. PDBTM's corresponding group is 17 unique pairs and 16.70% of records. The largest RCSB sequence pair contributes 3600 directional records across 13 PDBs. Thus a small set of low-performing, heavily repeated sequence pairs has a large effect on chain-instance weighting. Repetition itself is not proven to cause poor predictions.

## Prevalence and within-stratum contrasts

| Positive fraction | RCSB records | RCSB AP | PDBTM records | PDBTM AP |
|---|---:|---:|---:|---:|
| <=5% | 84075 | .14922 | 1160 | .28377 |
| >5-10% | 56792 | .26062 | 789 | .33101 |
| >10-20% | 71672 | .36507 | 3628 | .46445 |
| >20% | 35187 | .53157 | 2483 | .48704 |

Sparse interfaces (<=5% positives) account for 33.94% of RCSB records versus 14.39% in PDBTM. AP depends on prevalence, so raw AP across these groups is not a pure model-quality comparison. However, repetition-associated deficits persist within RCSB prevalence bins:

- >10-20% positives: singleton AP .49741 at prevalence .14630, versus >20-repeat AP .29035 at prevalence .13929.
- >20% positives: singleton AP .62935 at prevalence .30226, versus >20-repeat AP .47712 at prevalence .32342.
- <=5% positives: singleton AP .14899, >20-repeat .14842, showing no similar difference there.

This is descriptive adjustment with broad bins, not matched inference or proof of causality. Families, geometry, interface type and label-context multiplicity remain confounded.

## Other strata

Identical-sequence pairs have AP .28689 in RCSB; different-sequence pairs .29851. Identity class alone therefore does not explain the full deficit. It is not a comprehensive biological homomer/heteromer annotation.

| Chain length | RCSB AP | PDBTM AP |
|---|---:|---:|
| 100-299 | .29779 | .44143 |
| 300-599 | .29004 | .44817 |
| 600-999 | .20361 | .35127 |
| >=1000 | .29407 | .29280 |

Long chains are not the sole driver: the 600-999 group is only 4.26% of RCSB records, and deficits also occur in ordinary lengths. No <100 records were present in these evaluated manifests; the ge50 source-universe name does not imply such records survived the feature-ready training manifest.

## Updated training progress

All three HPC3 training jobs remain RUNNING, now in epoch6. Most recent complete epoch5:

| Run | Train loss | Val loss | Macro AP | Pooled AP | AUROC |
|---|---:|---:|---:|---:|---:|
| 1M | .657602 | .756929 | .295268 | .259492 | .762007 |
| 4M | .641505 | .769472 | .302805 | .262841 | .765891 |
| 8M | .649989 | .752389 | .307944 | .270712 | .764436 |

8M shows modest macro and pooled ranking gains versus its earlier epochs. This does not invalidate the native-validation deficit, and should not be interpreted as evidence that more epochs will automatically close it.

## Interpretation and next discriminating step

The concern about poorer native RCSB performance is supported by raw validation results. Three separable effects are now measured: probability overestimation, a much larger sparse-interface share, and heavy weighting of difficult repeated sequence pairs. The previous shared-label result still supports improved ranking on that selected intersection; it does not negate these full-cohort observations.

Next, inspect the high-repeat groups for differences in partner-specific interface masks under identical sequence inputs, assembly context, and predicted-monomer structure availability/quality. A subsequent controlled training comparison could use group-balanced sampling while retaining every structural-context label. Do not remove hard cases or replace the primary metric merely to improve the reported score. No such training change was made here. These are single-seed descriptive analyses, without new significance claims.
