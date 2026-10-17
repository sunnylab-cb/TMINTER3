# Formal RCSB pairwise training contract

Submitted on star as **job 2349**, normal QoS, using the reviewed launcher.
61 local tests passed and the narrow gpt-6-luna/max review cleared submission.
This supersedes smoke job 2348; it is a fresh run, not a smoke warm-start.

## Qualification

Star smoke job 2348 completed successfully (0:0) in 75 seconds on A100 80GB.
The one-epoch train+validation segment took 15.11 seconds for 256 training
pairs and 64 validation pairs. best.pt, last.pt and final_metrics.json exist.
Test was not read. Its low metrics are engineering-smoke results, not estimates
of the trained system's performance or full-length throughput.

## Frozen training configuration

`configs/experiments/20261016_rcsb/rcsb_s5_noESM2contact_full.yaml`.

- Model: ESMC 3456D + train-normalized Protenix spatial scalar89 + availability
  flag, TCN/global context, rank8 outer-product partner module, ~2.43M parameters.
- Explicit limitation: no RCSB ESM2 contact graph, not exact historical S5.
- Fresh initialization, seed49, 15 epochs. Do not warm-start from smoke weights.
- Entire feature-ready train set: 1150993 contacting pair contexts.
- Entire feature-ready validation set: 123863 pair contexts.
- Test remains sealed (evaluate_test=false); 131829 test pairs are present only
  in the manifest metadata, not evaluated or used to select models/gates.
- No smoke pair caps or 512-aa limit. Observed maximum train length is 4091;
  largest train matrix is 16613776 cells. Validation maximum is 3655 / 13359025.
- Batch cap64, padded-token cap65536, sum-of-pair-cell cap1000000, singleton
  cell cap25000000. Large pairs run alone; no silent truncation or dropping.
- Eight loader workers, pinning, prefetch2, bounded per-worker feature cache64.
- AdamW, LR1e-4, weight decay0.1, label smoothing0.05, ratio loss0.25, AMP.
- Validation directional-chain macro AP selects best.pt. The final selected
  checkpoint's validation predictions select the threshold. Repeated threshold
  sweeps are disabled during epochs; no test-based model or threshold selection.

Full split membership and feature-ready membership remain separate. The latter
excludes missing-feature endpoints and is recorded by ready_v1/excluded_pairs.csv
and missing_unique.tsv. This first run does not claim coverage of all 251267
eligible unique sequences, nor use experimental coordinates as input features.

## Runtime and recovery

Slurm launcher: scripts/slurm/star_rcsb_train_full.slurm. One GPU, eight CPUs,
128 GB RAM, normal QoS, 16-hour scheduling window. Critical's one-hour limit is
not used for the long formal run. Scheduling priority changes do not change the
training contract.

Every250 steps, atomic progress.pt stores model, optimizer, AMP scaler, Python/
NumPy/torch RNG states, epoch/batch cursor, global step, best-selection state and
loss histories. Model inputs and hyperparameters must match the saved config.
Each epoch has deterministic batching; resume skips completed batches without
loading their data or repeating their gradient updates. Loader worker seeding
uses a separate generator when checkpointing is enabled, so restarting workers
does not perturb model-dropout RNG. This assumes deterministic feature loading
without random augmentation, as in the current RCSB adapter.

USR1 is requested180 seconds before wall time. Training stops after an optimizer
step and saves; validation can be interrupted using the pre-validation progress
checkpoint. Exit75 asks the wrapper to requeue its own Slurm job. Preemption also
uses scheduler REQUEUE. The wrapper resumes from progress.pt; after more than32
restarts it stops for inspection. Other errors are not automatically retried.
No other user's jobs are manually cancelled.

CPU regression tests compare uninterrupted training with interruption during
training or validation followed by resume; final model tensors and validation
metrics must match exactly. This is not a claim of bitwise reproducibility
across arbitrary GPUs, drivers or nondeterministic kernels.

## Outputs

Star root:
`/data1/wholeset/RCSBprot/training/ppis_ge50_v1/`.

- runs/rcsb_s5_noESM2contact_full_seed49_v1/config.json
- runs/rcsb_s5_noESM2contact_full_seed49_v1/progress.pt
- runs/rcsb_s5_noESM2contact_full_seed49_v1/{best,last}.pt
- runs/rcsb_s5_noESM2contact_full_seed49_v1/metrics.jsonl
- runs/rcsb_s5_noESM2contact_full_seed49_v1/final_metrics.json
- logs/full_JOBID.log (append across requeues)

Training log: logs/full_2349.log. Training logs include epoch batch count, segment throughput and peak allocated
GPU memory. Estimate full-run ETA from real full-length training, not from the
short smoke cohort. At submission the job is pending GPU resources.
