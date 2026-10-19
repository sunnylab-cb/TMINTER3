# RCSB pairwise generalization diagnostic

Read-only audit on 2026-10-18. No training jobs or test evaluation changed.

## Sources

- HPC3 root: `/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1`
- Manifest: `training/ppis_ge50_v1/ready_v1/ready_pair_manifest.csv`
- Run: `_hpc3_training/20261017_v1/runs/large_batch8m/config.json`
- Loader: `src/ppcbind/data/rcsb_pair_dataset.py`, `_load_side` and `__getitem__`.

Only train and validation rows were aggregated; no test labels were analyzed.

## Manifest results

| Measure | Train | Validation |
|---|---:|---:|
| Pair instances | 1,150,993 | 123,863 |
| PDB entries | 97,071 | 14,514 |
| Unordered exact-sequence pairs | 268,690 | 22,274 |
| Same-sequence partners | 37.3868% | 60.1818% |
| Directional-chain macro positive fraction | 10.1627% | 10.9488% |
| Top ten PDB share of pair instances | 1.1540% | 6.7575% |
| Maximum multiplicity of a sequence pair | 12,250 | 1,800 |
| Sequence-pair groups with different positive-count tuples | 85,196 | 7,369 |
| Rows belonging to those heterogeneous groups | 923,192 | 106,171 |

For the final two rows, canonicalize left/right by sequence ID, swap positive
and observed counts accordingly, group by `(seq_left, seq_right,
observed_left, observed_right)`, and count distinct positive-count tuples.
This is a manifest-level heterogeneity screen, NOT a full label-vector audit.
Equal observed counts alone do not prove equal observed-mask positions.
The row counts are not counts of wrong labels or irreducible errors.

## Interpretation

The loader builds each side from unique-sequence ESMC and monomer structural
features. Chain IDs are zero and position indices start at zero. Assembly
instance identity and relative complex pose are not model inputs.
Therefore, different contacts between copies of the same sequence pair can
have indistinguishable inputs but different partner-specific labels.
This is a substantive hypothesis for the validation plateau, especially
given validation's higher same-sequence-partner fraction. Real alternative
interfaces are not necessarily labeling errors. Quantifying their effect
requires mask-aligned full label-vector aggregation and prediction analysis.

Raw loss comparisons also differ in positive weights (PDBTM 6.93964 versus
RCSB 10.38067). Evaluation averages batch losses equally rather than weighting
by directional-chain counts. Neither explains the AP plateau by itself.
Other confounds remain: no ESM2 contact graph in RCSB, Protenix versus
ESMFold descriptors, and different validation domains.

## Next discriminating checks

1. Audit identical-input label vectors on common observed positions.
2. Compare validation AP/loss for singleton versus repeated input pairs,
   and same-sequence versus different-sequence partners.
3. Report sequence-pair-macro metrics alongside instance-weighted metrics.
4. Apply the same audit to PDBTM S5 before attributing the difference to
   dataset expansion.
5. Preserve partner-specific labels; do not silently union alternative
   interfaces, which would change the task.

## Follow-up: PDBTM control falsifies a simple homomer explanation

The original S5 pair manifest on HPC2 contains exactly the run's 16,103
training and 4,030 validation pairs. Applying the same positive-count screen
using sequence SHA1 rather than chain IDs gives:

| Measure | PDBTM train | PDBTM validation |
|---|---:|---:|
| Same-sequence partners | 36.6019% | 80.1489% |
| Sequence-pair plus observed-count groups | 5,221 | 1,098 |
| Different positive-count groups | 1,871 | 567 |
| Rows in those groups | 12,194 | 3,464 |

Thus PDBTM validation has MORE same-sequence pairs than RCSB validation.
Rows in heterogeneous groups are approximately 86.0% in PDBTM validation
and 85.7% in RCSB validation. Input ambiguity exists, but these aggregate
statistics do not explain the performance gap. Do not present homomer
frequency as the established cause of RCSB's worse validation curve.

### Verified RCSB example

Validation PDB 10ac has three 402-residue identical-sequence chains. Label
records explicitly mark observed-sequence-only. For A-B, A has 36 positive
positions; for A-C, A has 37. These sets share only two indices (6 and 84;
zero based), giving intersection/union 2/71. The unique-sequence monomer
input for A and its partner is identical in both cases. This proves a
specific ambiguity, not its contribution to the global performance gap.

Prioritize controlled feature ablation and validation-domain comparison;
retain full-vector ambiguity analysis as a secondary stratified diagnostic.

## Feature-scale sampling

Deterministic sample: for each train/validation unique-sequence pool, sort
IDs by SHA256(`audit20261018` + UID), take the first 128. Load CPU tensors
from the actual HPC3 feature paths and apply the frozen normalization.

| Measure | Train | Validation |
|---|---:|---:|
| Sample sequences | 128 | 128 |
| Residues | 40,063 | 42,355 |
| Absolute normalized scalar p50 | 0.6111 | 0.6096 |
| p95 | 1.9939 | 1.9636 |
| p99 | 3.0215 | 2.9117 |
| p99.9 | 5.0944 | 4.5327 |
| Fraction of scalar elements above absolute 10 | 0.00013462 | 0.00010426 |
| Largest absolute per-channel sample mean | 0.1257 | 0.0856 |
| Mean per-sequence embedding standard deviation | 0.04378 | 0.04390 |

All sampled normalized structural tensors are finite. The model sanitizes
structural features with clipping at 10 before projection. This sample does
not indicate gross train/validation normalization drift; it is not an
exhaustive outlier or feature-semantic audit.

An old S5 ESMC sample (`8wtp__A.pt`, 614 x 3456) has standard deviation
0.04415, same layers [12,24,36], final-layernorm, and revision
`e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c`. Its similar scale does not prove
tensor equivalence, but argues against a gross old/new embedding scale error.
Frozen structural normalization metadata records train-only fitting on
170,465 unique sequences / 55,003,867 residues, with population variance.

## Loss reference: fixed uninformative prediction

No fitting or threshold search: set every residue probability to 0.5.
For chain positive fraction r and positive weight w, the implemented weighted
CE with smoothing 0.05 equals
`ln(2) * (0.95 + 0.05*(1+w)/(2*(1+(w-1)*r)))`.
Add ratio penalty `0.25*(0.5-r)^2`. Compute both directional-chain mean
and the mean of batch means using the actual PairTokenBatchSampler class.
RCSB class was extracted from frozen deployment source; PDBTM class was
read from the current HPC2 trainer (historical sampler identity is not yet
hash-verified).

| Validation cohort/profile | Constant pair-macro loss | Constant batch-mean loss | Batches |
|---|---:|---:|---:|
| PDBTM S5 | 0.767647 | 0.788162 | 836 |
| RCSB 1M | 0.815048 | 0.827612 | 14,835 |
| RCSB 4M | 0.815048 | 0.830689 | 3,794 |
| RCSB 8M | 0.815048 | 0.834247 | 2,003 |

Epoch-3 observed S5 loss 0.673885 improves 0.114277 over its constant
reference; RCSB 1M loss 0.759713 improves 0.067899. Their raw difference
0.085828 decomposes arithmetically into constant-reference difference
0.039450 plus difference in improvement 0.046378. This is descriptive,
NOT causal attribution or proof that 46% of performance loss is explained.
It supports both conclusions: raw losses are not directly comparable,
and the new model has not matched the old model's improvement over this
simple reference. Constant 0.5 is not the optimal constant predictor.

Validation directional-chain positive-fraction medians differ substantially:
PDBTM 0.175758 versus RCSB 0.085427. Macro means are 0.155845 and
0.109488 respectively. This further cautions against interpreting AP
differences as architecture/data-size effects on a common target.

## Frozen inference probe submitted

HPC3 Slurm job 734491 (`acd_u`, one GPU, 30 minutes) was observed RUNNING
on ACD1-16. It uses an isolated copy of large_batch4m/best.pt under
`_hpc3_training/20261017_v1/diagnostics/frozen_20261018/best4m.pt`.
The script records the loaded checkpoint SHA256 and epoch in its output.

`scripts/training/diagnose_rcsb_frozen.py` compares full output, disabled
partner delta, and zero normalized structural descriptors (availability
retained). It selects 256 validation pair instances by deterministic hash,
restricted to both chain lengths <=1024. Outputs are directional-chain-macro
AP, unweighted CE, and probability perturbations, with homo/heteromer strata.
These interventions can expose reliance on inputs but are not retrained
ablations and cannot establish the causal benefit of a feature.

Narrow reviewer findings about schema, empty labels and negative counts
were addressed before submission. Python compilation passed. Results remain
pending at submission; neither training runs nor test evaluation were modified.

### Completed probe

Job 734491 completed with exit 0 in 2m21s. Snapshot epoch 3, SHA256
`c5d8cabb06429b1ce064fae3d6bbd022dbea679ec29ba5cb574fbf9eb4eb8bdd`.
Eligible validation pairs: 122,497; selected 256 pairs / 512 directional
chains / 225 PDBs. Local raw artifact:
`artifacts/diagnostics/20261018/frozen4m_epoch3.json`.

| Intervention | Macro AP | Unweighted macro CE |
|---|---:|---:|
| Full model | 0.288602 | 0.669186 |
| Partner delta disabled | 0.275963 | 0.640862 |
| Normalized PS set to zero | 0.220479 | 0.698414 |

Paired PDB-cluster bootstrap, 10,000 resamples, seed 20261018. Resample
whole PDB groups and calculate sum of chain-level differences divided by
resampled chain count, preserving the chain-macro estimand:

- Full minus no-partner AP: +0.012638, CI [0.001611, 0.023753].
- Full minus zero-PS AP: +0.068123, CI [0.055080, 0.081919].
- Full minus no-partner unweighted CE: +0.028324,
  CI [0.015377, 0.042389] (higher CE is worse).

PDB bootstrap does not account for all cross-PDB homology dependencies;
these are exploratory intervals, not definitive significance claims.
The structural path is used and useful to this frozen model's ranking.
Partner correction has a modest ranking benefit but worsens unweighted
log loss; calibration is a hypothesis, not established by CE alone.
Disabling a jointly trained branch is not equivalent to training without it.
These results do not isolate ESM2 graph removal or prove the cause of the
old/new performance gap. A true partner replacement diagnostic is still
needed to distinguish partner-specific information from extra capacity.

### Partner replacement follow-up

Submitted job 734538 on acd_u (one GPU, 30 minutes); observed pending
Priority. Uses exactly the same snapshot and hash-selected 256 query pairs.
For each direction, choose the nearest-length partner from all length-eligible
validation rows, excluding the query PDB and both original sequence hashes
(case normalized). Keep query labels unchanged. This measures counterfactual
sensitivity, not accuracy on biological negative pairs. Source was reviewed
for correct-side logits, exclusions and metric aggregation; exclusion toy
tests and compilation passed before deployment.
Remote outputs: `diagnostics/frozen_20261018/replacement4m/summary.json`.

Replacement follow-up finished successfully. The original three intervention
metrics exactly reproduce the prior run. Correct-partner macro AP is
0.288602 versus replaced-partner 0.277589, difference +0.011013;
PDB-cluster bootstrap CI [0.003417, 0.018994] (10,000, seed 20261018).
Mean absolute probability change is 0.049208. Homomer AP decreases
0.278704 -> 0.267336; heteromer AP 0.303546 -> 0.293068.
99.414% of directional replacements have exactly matching lengths;
maximum length difference is three residues. Unweighted CE decreases
0.669186 -> 0.624823 after replacement; this does not contradict optimization
of a positive-weighted training objective and is not standalone calibration
proof. The same original labels are used only as a sensitivity reference,
not as ground truth for the artificial pair.

This rejects a completely partner-blind explanation on the sampled cohort,
but shows a modest partner-identity contribution compared with structural
descriptor sensitivity. It does not establish why the PDBTM/RCSB gap exists.
Only one deterministic donor selection and one seed/epoch were tested.
Artifact: `artifacts/diagnostics/20261018/replacement4m_epoch3.json`.

## Matched PDBTM graph control

Submitted HPC2 array 13179672 (tasks 0 graph, 1 no_graph), partition
`i64m1tga800u`, one GPU / 8 CPU / 128 GB / 12h per task. Both start from
scratch, seed49, original S5 20-epoch LR schedule and data; test disabled.
The original S5 config is deep-copied; only use_contact_graph, output path
and descriptive metadata differ. A fresh graph-enabled replica controls
for current code/runtime differences rather than relying solely on the old run.
Common data paths remain required even when graph is disabled.
CPU model construction verified parameter counts 2,756,623 / 2,426,895
and existence of all data paths. Config invariants and narrow review passed.
Training output creation rejects existing directories; no resume is used.

Frozen deployment:
`/hpc2hdd/home/shuang886/jhspoolers/PPC/experiments/tminter3_s5_graph_control_20261018`
Run directories:
`runs/pdbtm_S5_graph_control_{graph,no_graph}_seed49_20261018`.
This is a single-seed architecture control, not a proof of data-expansion
benefit on a shared holdout. Different parameter counts can also change
initialization RNG consumption, despite identical seeds.

## Shared-validation feasibility audit

Old S5 validation contains 1,011 PDB IDs and 1,061 exact sequence hashes.
New RCSB TRAIN overlaps 832 of these PDBs and 760 exact sequences;
new RCSB VAL overlaps 103 PDBs and 139 exact sequences. This is overlap
between two experiments' different splits, not evidence of leakage inside
either split. Never evaluate expansion benefit on the entire old validation
cohort and call it independent for the RCSB model.

Filtering old validation pair rows to PDB IDs in new validation AND both
sequence SHA1s in new validation yields 352 pair instances / 100 PDBs /
139 unique sequences. This is a candidate common validation cohort for
same-label comparisons. It is a selected intersection, not representative
of all RCSB, and still needs materialized manifest/hash audit and feature
availability checks before inference. Only train/validation identities were
read; no test labels were used.

Shared cohort was exported on HPC2 as
`experiments/tminter3_s5_graph_control_20261018/shared_validation.json`:
352 unique pair IDs / 100 PDBs. Export checks original label metadata,
5.5 A/heavy-only definition, sequence SHA1, label values before casting,
lengths, observation masks, positive counts, and old-training exact exclusion.
Independent new-training audit found zero shared-identity sequence hashes
and zero shared-identity PDB IDs in new train.

## Graph control epoch 1

| PDBTM model | Train loss | Val loss | Macro AP | Pooled PR AUC |
|---|---:|---:|---:|---:|
| Graph replica | 0.677468 | 0.684627 | 0.438186 | 0.362111 |
| No graph | 0.677588 | 0.678506 | 0.435774 | 0.366897 |

Graph replica closely reproduces historical S5 epoch1 (macro AP 0.438385,
val loss 0.684799), providing an empirical runtime/code sanity check.
Removing graph changes epoch1 macro AP by -0.002412 and improves val loss
by 0.006122. Thus graph absence does NOT explain the large initial RCSB gap
on its own. This is only epoch1 of a single seed, not the final ablation.
Copies of current last.pt were saved as graph_epoch1.pt/no_graph_epoch1.pt
in the control deployment; checkpoint epoch fields still require verification
before using those snapshots in shared-cohort evaluation.

## Strict shared-pair cohort and epoch-1 evaluation

Further association audit excluded `6ivn__A--6ivn__E`: its PDB and hashes
were individually in new validation but the PDB/sequence-pair association
was not. Fixed cohort now has 351 pairs / 100 PDBs; the original 352 export
is retained. `shared_validation_matched_pairs.json` records the exclusion
and parent artifact hash. No prediction metrics were used for this selection.

Verified both PDBTM snapshots and RCSB full best snapshot have epoch=1.
`evaluate_shared_validation.py` checks checkpoint epoch, same cohort labels,
PDB-plus-sequence-pair validation association, train identity exclusion,
sequence/label lengths and counts, and strict checkpoint loading. Each
system uses its own frozen feature pipeline. Metrics use common unweighted
CE and AP/AUROC without threshold search. Label definition is the original
PDBTM pairwise labels for all three, not native new-assembly labels.

Submitted jobs:
- HPC3 734693: RCSB epoch1.
- HPC2 13180781: PDBTM graph epoch1.
- HPC2 13180782: PDBTM no_graph epoch1.

Output directories: HPC3 diagnostic `shared_rcsb_epoch1`; HPC2 control
deployment `shared_graph_epoch1` and `shared_no_graph_epoch1`.
The script writes per-directional-chain records and aligned probability
arrays for paired analysis. This selected common cohort cannot establish
whole-RCSB generalization or isolate ESMFold versus Protenix effects.

## Shared epoch-1 results: ranking gap reverses

All three outputs have identical cohort SHA256
`2f3a524b9347c9f5f414856498f905af42201682b5fd1226ca310f38fe367abd`,
351 pairs, 371,113 observed directional residues, prevalence 0.1008345.

| System | Macro AP | Pooled AP | AUROC | Unweighted macro CE |
|---|---:|---:|---:|---:|
| PDBTM graph | 0.363596 | 0.310493 | 0.808023 | 0.562906 |
| PDBTM no_graph | 0.377832 | 0.320286 | 0.808383 | 0.542956 |
| RCSB no_graph | 0.409249 | 0.340468 | 0.842578 | 0.604601 |

Paired PDB-cluster bootstrap, seed20261018, 10,000 draws; chain-weighted
mean differences after resampling 100 PDB groups:
- RCSB minus PDBTM graph AP: +0.045653, CI [0.027918, 0.063259].
- RCSB minus PDBTM no_graph AP: +0.031417, CI [0.016839, 0.046653].
- RCSB minus PDBTM graph macro CE: +0.041695, CI [0.028051, 0.054932].
- RCSB minus PDBTM no_graph macro CE: +0.061645, CI [0.046015, 0.076596].

Cross-PDB homology dependence is not modeled by these exploratory intervals.
Raw records are `artifacts/diagnostics/20261018/shared_{rcsb,graph,no_graph}_epoch1.json`.
On this fixed shared cohort, RCSB is better at ranking, not worse. Its
probability loss is worse. Separate-cohort AP comparisons therefore did
not establish degradation from expanding training data. The selected
intersection is not a full-RCSB benchmark; features and label-generation
sources still differ between systems, and equal epochs are not equal updates.

## Probability-scale diagnostic

On the exact shared labels, pooled statistics are:

| System | Mean predicted p | Prevalence | Pooled CE | Brier | ECE15 |
|---|---:|---:|---:|---:|---:|
| PDBTM graph | 0.330845 | 0.100835 | 0.486184 | 0.156156 | 0.230011 |
| PDBTM no_graph | 0.321711 | 0.100835 | 0.465477 | 0.147489 | 0.220876 |
| RCSB no_graph | 0.379887 | 0.100835 | 0.517167 | 0.168313 | 0.279063 |

For an exploratory, training-parameter-only inverse, use
`t=q/(w-(w-1)*q)` then `p=clip((t-0.025)/0.95,1e-7,1-1e-7)`, where w is
the actual class weight and smoothing is 0.05. No validation fitting occurs.
This is the idealized weighted-CE inverse; per-chain loss normalization,
ratio penalty, finite model capacity, regularization and domain shift mean
it is NOT guaranteed calibrated for this model. Clipping can introduce ties.

| System after diagnostic inverse | Mean p | Pooled CE | Brier | ECE15 |
|---|---:|---:|---:|---:|
| PDBTM graph | 0.094949 | 0.360247 | 0.081829 | 0.044955 |
| PDBTM no_graph | 0.085663 | 0.356168 | 0.081029 | 0.042742 |
| RCSB no_graph | 0.067973 | 0.311661 | 0.077273 | 0.032862 |

The marked improvement supports class-weight/smoothing-induced probability
inflation as an important contributor to unweighted probability loss. The
RCSB inverse underestimates prevalence, so this is diagnostic evidence,
not a recommended final calibration or threshold replacement. No frozen
prediction files or checkpoints were altered.

## Epoch 3 confirmation and diagnostic conclusion

Shared-cohort jobs: HPC3 734704, HPC2 13181174/13181175.

| System, epoch 3 | Macro AP | Pooled AP | AUROC | Unweighted macro CE |
|---|---:|---:|---:|---:|
| PDBTM graph | 0.365030 | 0.303292 | 0.798687 | 0.517478 |
| PDBTM no_graph | 0.358581 | 0.295862 | 0.796821 | 0.520831 |
| RCSB no_graph | 0.423097 | 0.336430 | 0.842723 | 0.573215 |

RCSB minus graph macro AP: +0.058068, exploratory PDB-bootstrap CI
[0.042174, 0.074364]. RCSB minus no_graph: +0.064516,
CI [0.046989, 0.082489]. RCSB epoch3 minus epoch1 macro AP:
+0.013848, CI [0.003036, 0.025936]. Its macro CE decreases by
0.031386, CI [-0.046248, -0.016189]. Pooled AP decreases slightly
0.340468 -> 0.336430, so not every metric improves over time.

All six evaluations (three systems, epochs1/3) have identical flattened
observed-label SHA256 (little-endian int64):
`f9ec2a5d9a853ac73b5525fc89d864f19712c99973e6ab905fa0df7040e6b119`;
directional-chain length SHA256:
`acf380bae895c852060fb0be46e32ffdf8c36c22f7c5a98bf2a54fd37adbb4ea`.
Per-record IDs and positive fractions also match. Before collating, each
backend's label vector is checked against the same export, including -100
mask positions. Sequence hashes tie feature order to that exported sequence.

### What the evidence answers

1. The original premise mixed different validation domains and differently
   scaled objectives. Native losses/AP do not establish poorer RCSB learning.
2. On the fixed shared labels, RCSB ranking is better at both epochs1 and3,
   and chain-macro ranking improves between those epochs. This contradicts
   a blanket claim that expansion already destroyed prediction quality.
3. Higher unweighted probability loss remains real on the common cohort.
   Strong positive-probability inflation and its training-weight inverse
   diagnostic support a probability-scale/objective mismatch explanation.
   The inverse is not an approved production calibration procedure.
4. Early graph-controlled training does not show graph removal causing a
   large deficit. No gross numeric-feature failure was found in the sample.
   Partner sensitivity exists but is modest. Input ambiguity exists in both
   datasets and its frequency does not explain their performance gap.
5. Native full-RCSB validation plateau is still real. Shared-cohort gains do
   not establish improvement across its remaining nonshared domains. The
   source domain/labels, feature sources, class weights, and exposure budgets
   differ; no isolated causal effect of training-set size is claimed.

The same-epoch comparison is under each run's own training protocol, not
equal optimizer updates: 2,004 train batches/epoch for S5 versus 133,609
for RCSB 1M (about 66.7x). Both use seed49, but one seed is not a stability
study. PDB-level intervals omit cross-PDB homology dependence. Common
validation is a selected 100-PDB intersection, not a fresh sealed holdout.

### Practical next decisions, beyond this diagnosis

- Keep ranking metrics and probability losses separate; do not select based
  on raw cross-dataset weighted loss.
- Retain chain-macro unweighted CE and sample-weighted loss reporting in
  future evaluations, alongside the optimized weighted objective.
- If calibrated probabilities/ratios are required, use a separately reserved
  calibration partition or nested component-level validation, not a threshold
  tuned on this repeatedly inspected diagnostic cohort.
- Before further architecture sweeps, stratify native RCSB validation by
  domains/interface classes; the shared-subset result does not justify
  assuming every domain benefits.
- Existing graph-control training continues for its allocated 20 epochs;
  later results are supplementary, not required to establish the early-epoch
  conclusions above. No training job was stopped or test set evaluated.
