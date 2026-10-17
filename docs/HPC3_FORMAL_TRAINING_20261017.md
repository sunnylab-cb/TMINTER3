# HPC3 full-RCSB pairwise training deployment

## Scope

PPIS only. Original star full-seed49 frozen source is reused without model,
split, label, normalization, or optimizer changes. No existing star jobs or
HPC3 git worktree files are changed. Training starts from scratch; smoke
weights are never used for formal initialization.

Data root:
`/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1`

Deployment relative to data root: `_hpc3_training/20261017_v1/`.
Code is extracted into `code/`; configs in `configs/`; outputs in `runs/`.
Python: `/data/user/shuang886/.conda/envs/tminter3_esmc/bin/python`.

## Frozen contract

- 1,150,993 training pairs; 123,863 validation pairs; 131,829 sealed test pairs.
- 214,097 feature-ready exact sequences across all three splits.
- ESMC layers 12/24/36, 3456D; predicted-structure scalar89 + availability = 90D.
- TCN single-chain encoding plus outer-product pair head, rank8, hidden32.
- No ESM2 contact graph. This is explicitly `RCSB_S5_noESM2contact`.
- Seed49, 15 epochs, AMP, batch64, 65,536 token budget, 1M batch pair-cell budget.
- Validation directional-chain macro AP selects the checkpoint. Test is not evaluated.
- Full config differs from frozen star YAML only in five data paths and output path.

Ready manifest SHA256:
`4d908b57ecba298697852d0ed808c9ed48989dfd77439db42b0f97f2f7d31fa8`

Original archive SHA256:
`2f2cc672453bed0d4554b92ae21b8caa1f58ab31c631087bc068d959816ed606`

`deployment.json` records extracted code/config hashes. Launcher verifies hashes
on every invocation, including resume. Checkpoints are trusted, job-generated
local artifacts; they are not accepted from third parties.

## Jobs

- Smoke: `728951`, acd_u, 1 H100 80GB, 8 CPU, 128GB RAM, 30min.
  Full manifest and train normalization; explicit caps256 train/64val, length512,
  one epoch. These caps are not present in the formal config.
- Formal: `728963`, acd_u, 1 GPU, 8 CPU, 128GB RAM, 48h per allocation;
  dependency `afterok:728951`, kill on invalid dependency.
- Formal startup checks successful smoke final metrics, finite AP, sample counts,
  config agreement, finite best/last checkpoint tensors, and sealed-test status.
  It then writes `SMOKE_APPROVED.json`. A failure prevents training.
- Progress checkpoints every250 steps; original signal/exit75 requeue behavior
  retained, maximum32 restarts. Final status is recorded in `final_metrics.json`.

Smoke completed with exit0:0 in2min38s. Epoch train/validation wall time was
13.9147s, train loss0.812258, validation loss0.814237, validation macro AP0.140934,
swap-symmetry maximum logit error0.0. Test was not read. These small-sample
scratch smoke metrics are engineering checks, not model performance claims.
Formal job728963 subsequently entered RUNNING on ACD1-19. Full training is
not complete; throughput and ETA require measurements from the full run.

## Verification

Seven focused local tests pass: path-only rebase, no smoke-cap propagation,
invalid seed/epoch/test/init/cap rejection, and missing-smoke fail-closed check.
Python parsing and shell syntax pass. Narrow gpt-6-luna max review prompted
archive digest pinning, explicit seed/epoch assertions, startup hash validation,
and inclusion of the checker in the deployment.
