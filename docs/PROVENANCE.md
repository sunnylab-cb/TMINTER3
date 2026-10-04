# Source provenance

Repository assembled on 2026-10-04 from the PDBTM partner-specific experiments.

- Parent project: https://github.com/HUSRCF/PPC
- Parent HEAD observed on HPC2: 7df905069d59e6c510f7148b35daaccc1b4d6b50
- Pairwise source files were untracked in that parent project; the parent
  commit alone does **not** reproduce these experiments.
- Executed follow-up snapshot:
  `/hpc2hdd/home/shuang886/jhspoolers/PPC/experiments/pairwise_followup_20261002`
- Core sources match that snapshot; see `SOURCE_SHA256.txt`.
- Original experiment bundle remains in
  `/home/husrcf/Code/ProtBind/pypropel_unibio/tmp/pairwise_hpc2_20261001`.

Extraction changes: repository documentation and ignore rules; relocate
diagnostic beside its training dependency; repair the loss-test relative
path; add a configurable Slurm entry point. Model and training numerics were
not changed by this extraction. Historical Slurm scripts preserve original
absolute paths and are archived rather than advertised as portable.

Original Slurm jobs: S0-S4 12903274-12903278; S5-S9 12904487-12904491;
partner diagnostic12904492; follow-up smoke12904482.

No LICENSE file was found at the parent workspace root. This extraction does
not introduce a new license grant. Confirm project licensing before external
redistribution under a claimed open-source license.
