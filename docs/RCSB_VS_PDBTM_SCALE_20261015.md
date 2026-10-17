# RCSB versus original PDBTM scale

This compares data record scale, not independent biological interactions or
equivalent label/assembly coverage. It does not compare predictive performance.

| Unit | Original PDBTM | RCSB | Ratio |
| --- | ---: | ---: | ---: |
| Exact unique observed sequences, all lengths | 6667 | 271397 | 40.71x |
| Chain instances, all lengths | 20240 | 1072528 | 52.99x |
| Exact unique sequences, >=50 aa | 6659 | 251267 | 37.73x |
| Chain instances, >=50 aa | 20222 | 988743 | 48.89x |
| Contacting chain-pair records, before the new >=50 gate | 23196 | 1903031 | 82.04x |

Both pair counts refer to one undirected chain-pair context with labels for
both directions in the same record, not two separately counted directions.
RCSB source `build_rcsb_pair_tags.py` constructs one pair record for each
contacting chain-instance pair within an expanded biological assembly.

Important scope differences:

- Original PDBTM sequence/chain counts are the broader frozen eligible chain
  manifest, not a verified endpoint census of only contacting pairs.
- RCSB sequence/chain counts are from participating endpoints of contacting
  pair contexts. Neither unique count is a count of training examples.
- PDBTM pair labeling uses existing coordinate features; its generator does
  not itself reconstruct biological assemblies. The new RCSB labeling does.
  Assembly multiplicity can inflate the pair-record expansion relative to
  sequence diversity; 82.04x is not 82.04x independent interaction information.
- Both definitions use partner-specific 5.5 A heavy-atom contacts. New final
  counts must enforce >=50 aa on BOTH partners, then report missing-feature
  exclusions separately. Do not reuse 1903031 as that final eligible count.
- PDBTM train/val/test contacting pairs: 16103 / 4030 / 3063. Those split
  fractions are not the new RCSB 80/10/10 sequence-component target.

## Evidence

Local original label summary:
`/home/husrcf/Code/ProtBind/pypropel_unibio/tmp/pairwise_hpc2_20261001/results/label_generation/summary.json`

Original frozen chain manifest:
`/media/990Pro/ProtBind/PPC/features/contact_labels_5p5A/splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708/pypropel_chain_filtered_no_len_limit/chain_manifest.csv`

HPC2 new RCSB root:
`/hpc2hdd/home/shuang886/jhspoolers/PPC/experiments/tminter3_rcsb_20261009`

- `manifests/full_success_v1/summary.json`
- `manifests/observed_ge50_20261015/summary.json`
- `artifacts/mmseqs_ge50_20261015/ppis_only_complex_audit_v1.json`

The raw local snapshot contained 244406 mmCIF entries; 242944 were successfully
processed, and 146464 entries contributed interacting chains to the frozen
manifest. These snapshot counts are not a live census of all RCSB deposits.
