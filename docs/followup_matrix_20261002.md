# PDBTM S5-S9 follow-up deployment

All variants use the S1 scratch contract: seed49,20epochs, validation
directional-chain macro AP checkpoint selection, no test loading, frozen
split and predicted feature sources. No checkpoint warm-start or resume.

| Variant | Single change vs S1 |
|---|---|
| S5_site_only | pair_contact_loss_weight 0.2 -> 0 |
| S6_no_ratio | ratio_loss_weight 0.25 -> 0 |
| S7_no_smoothing | label_smoothing 0.05 -> 0 |
| S8_endpoints | append projected endpoint sum, absolute difference, product |
| S9_null_pool | length-normalized null-aware softmax pooling + real-mass input |

S8 retains rank8 and32 output pair channels. It tests endpoint identity
information beyond the symmetric product, not a full-width PLMGraph-Inter
reproduction or a strictly parameter-matched control (projection gains768
weights plus48 normalization parameters).

S9 real-candidate logits are contact_logit/temperature-log(partner_length).
A learned scalar null logit starts at0, with zero null context. Pooled context
and contact evidence are weighted by the non-null probabilities; their total
mass is also supplied to the site head. All zero contact/null logits give
mass0.5 at every partner length. This mass is an architectural gate, not a
calibrated probability that any physical contact exists.

## Diagnostic

S0 and S1 frozen best checkpoints are evaluated on validation only. Correct
partners are compared to replacements sampled with replacement from the64
nearest-length eligible validation chains. Candidates exclude the source PDB,
original partner sequence and query sequence; a pair-cell budget is enforced.
Seeds11/29/47 are fixed before seeing results. Record actual length mismatch,
per-direction AP/probability changes, homo/heteromer and PDB-macro AP drops.

This is a counterfactual sensitivity test with unchanged query targets, not
evaluation on verified noninteracting pairs. No replacement labels or contact
loss are invented. S0 should have exactly zero change. Encoder outputs are
cached once; first-batch original-forward reconstruction is verified.

## Isolation and tests

Remote source snapshot:
`/hpc2hdd/home/shuang886/jhspoolers/PPC/experiments/pairwise_followup_20261002`

Original S3/S4 source files and runs are untouched. Existing training script
SHA256: `86902aabc47696b9945a2a081f2d80a21af8cfeff4ef0ed8c48d15cd403cb0d8`.

CPU adapter tests passed: zero-init residual, swap symmetry, finite gradients,
null length normalization and negative/positive evidence limits. New configs
have unique output directories reserved atomically by the existing trainer.
GPU smoke covers all five training variants and S0/S1/S8 diagnostics, before
dependency-gated formal runs.12h Slurm limit and5s GPU telemetry are retained.

Requested reviewer gpt-5.3-codex-spark was rejected as unavailable by the
backend. Narrow review uses gpt-6-sol xhigh instead; this is not Spark.

## Submission

Review completed with no remaining launch blocker. Existing S0/S1 smoke
checkpoints were verified on HPC2. Diagnostic paths include Slurm job ID;
failed attempts remain intact. A failed training smoke must be retried with
new run directories, never overwritten or silently resumed.

| Job | Purpose |
|---|---|
|12904482|Five-variant GPU smoke plus three diagnostic smoke checks|
|12904487|S5_site_only|
|12904488|S6_no_ratio|
|12904489|S7_no_smoothing|
|12904490|S8_endpoints|
|12904491|S9_null_pool|
|12904492|S0/S1 full-validation partner replacement diagnostic|

All six downstream jobs depend on afterok:12904482 with
kill-on-invalid-dep=yes. At final submission verification the smoke was
PENDING(Priority), with no predicted start time; downstream jobs were
PENDING(Dependency). Thus CPU tests and review passed, but GPU smoke and
formal training have NOT yet been verified as running.

Training outputs: `runs/pairwise_followup_<variant>_seed49_20261002_hpc2`.
Diagnostic outputs: `runs/pairwise_followup_partner_diagnostic_<S0/S1>_12904492`.
Latest existing S3/S4 jobs continued untouched during deployment.
