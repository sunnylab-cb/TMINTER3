# Full-RCSB partner-specific PPIS model matrix

## Scope and contract

This is PPIS-only work, with no COACH420/Holo4k or ligand-site exclusions.
The active reference is job 2349, `RCSB_S5_noESM2contact`, not the historical
PDBTM S5 with ESM2 contacts. All runs start from scratch.

All five systems use the same frozen feature-ready manifests: 1,150,993 train
pairs, 123,863 validation pairs, and 131,829 sealed test pairs. The joint
MMseqs SI30/cov80 plus same-PDB split and train-only PS normalization are fixed.
This split does not claim exhaustive Biopython global-SI certification.

Inputs are frozen ESMC layers 12/24/36 (3456D) and Protenix Mini monomer
89 scalar descriptors plus availability (90D). Experimental inter-chain
contacts are supervision only, never inference features.

All runs share seed 49, 15 epochs, batch size 64, 65,536 token budget,
1,000,000 pair-cell batch budget, 25,000,000 single-pair cell ceiling,
8 loader workers, identical optimizer/schedule and checkpointing. No new
length or pair-count cap is introduced. These are epoch/sample-budget matched,
not FLOP-, parameter-count-, or wall-time-matched experiments.

## Questions and matrix

| ID | Model | Difference from reference | Question |
|---|---|---|---|
| R0 | Outer product, softmax | Running reference, job 2349 | Full-RCSB reference |
| R1 | Independent monomer predictor | No partner conditioning in forward | Does knowing the partner help with partner-specific labels? |
| R2 | Cross-attention | Shared symmetric residual; partner pooled into at most 32 sequence segments | Does this interaction operator outperform residue-pair outer products? |
| R3 | Outer product, null-softmax | Add a length-normalized null/no-match choice | Does forced matching to a partner hurt? |
| R4 | Outer product + contact supervision | Add 0.1 times balanced positive/negative pair-contact loss | Does explicit residue-pair supervision improve site prediction? |

R1 still receives partner-specific labels, not all-partner-union labels. The
same chain can have different labels for different partners; the inability
to resolve these is the intended no-partner control. Its registered but unused
adapter is not active in forward and should not be counted as active capacity.
R2 is segment-pooled cross-attention, not full residue-resolution attention.
R3 and R4 each change one mechanism; neither changes input data.

Selection: best validation directional-chain macro AP. Thresholds are selected
only on validation after checkpoint selection. `evaluate_test: false` on every
run. Compare macro AP, pooled AUPRC/AUROC, ratio error, learning curves,
throughput, peak allocated VRAM, and swap symmetry on validation. No test-based
run selection. Seed replication follows validation screening; a single seed
cannot establish a robust architecture improvement.

## Execution and cost

Use a separate immutable-at-launch copy of the reference source archive:
`/data1/wholeset/RCSBprot/training/ppis_ge50_v1/launch_records/full_seed49_v1/code_at_submit.tar.gz`.
Its SHA-256 is
`2f2cc672453bed0d4554b92ae21b8caa1f58ab31c631087bc068d959816ed606`.
Add only new matrix configs and launcher; do not change the live baseline tree.

Matrix snapshot destination:
`/data1/wholeset/RCSBprot/training/ppis_ge50_v1/launch_records/matrix_20261017_v1/code`.
Resolved configs and their hashes are in `configs/experiments/20261017_rcsb_matrix/manifest.json`.
Outputs: `.../runs/rcsb_matrix_<variant>_full_seed49_v1/`.
Logs: `.../logs/matrix_<array_job>_<task>.log`.

Slurm: normal QoS, one GPU per task, array 0-3 with concurrency at most 2,
8 CPUs, 128 GB RAM, 16-hour allocations. Every 250 steps save progress;
USR1 before time limit triggers checkpoint/requeue of that array element.
Unexpected failures, including OOM, remain failed for inspection rather than
silently shrinking batches or excluding samples. No manual preemption.

Reference observation before submission: epoch 1 step 17,825/133,609, about
19.4 pairs/s and 14.37 GiB peak allocated memory. At this rate a training epoch
is roughly 16-17 hours and 15 epochs roughly 250 GPU-hours, before validation
and queue time. This is a rough projection from partial-epoch throughput,
not an ETA for other operators. Four extra jobs could consume order 1,000
GPU-hours if similarly paced. Long-chain OOM/performance remains unverified
for the full epoch even though fixture tests pass.

Do not increase the reference batch mid-run. Any throughput optimization must
be profiled separately and distinguished from this controlled matrix. Joint
Pairformer is deferred until long-chain memory/cubic compute is tested; do not
silently shorten its input universe to make it fit.

## Checks

Production-width CPU fixture training/validation passes for all four variants;
R4 has nonzero contact loss. Config tests ensure all data/base-model/training
fields are unchanged except declared pair mechanism, contact-loss weight and
output directory. These synthetic tests verify execution, not predictive
performance or large-GPU memory safety. Shell syntax checked. Independent
review requested from gpt-6-luna at max reasoning.

Submission IDs and final test counts are recorded in
`refine-logs/EXPERIMENT_TRACKER.md` after submission.
