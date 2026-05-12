# Pharos Global PIQA EU Subset

This directory contains a subset of the Global PIQA benchmark, specifically filtered for EU-relevant languages (official EU languages + languages with significant presence in EU member states) as part of the Pharos project.

## Paper

See the original task: [Global PIQA](../../global_piqa/README.md) for more information.

## Pharos EU Subset Details

This subset includes both evaluation methods provided by the original benchmark:

### Groups

- `pharos_global_piqa_eu`: Aggregator group including both variants below.
- `pharos_global_piqa_completions`: Multiple-choice (log-likelihood) variants for EU languages.
- `pharos_global_piqa_prompted`: Instruction-following (generation) variants for EU languages.

### Included Languages (20)

The following 20 official EU languages are included:

| ID | Language              | Global PIQA Code |
| :- | :-------------------- | :--------------- |
| 1  | Bulgarian             | `bul_cyrl`       |
| 2  | Croatian              | `hrv_latn`       |
| 3  | Czech                 | `ces_latn`       |
| 4  | Dutch                 | `nld_latn`       |
| 5  | English               | `eng_latn`       |
| 6  | Estonian              | `ekk_latn`       |
| 7  | Finnish               | `fin_latn`       |
| 8  | French (France)       | `fra_latn_fran`  |
| 9  | German                | `deu_latn`       |
| 10 | Greek                 | `ell_grek`       |
| 11 | Hungarian             | `hun_latn`       |
| 12 | Italian               | `ita_latn`       |
| 13 | Lithuanian            | `lit_latn`       |
| 14 | Polish                | `pol_latn`       |
| 15 | Portuguese (Portugal) | `por_latn_port`  |
| 16 | Romanian              | `ron_latn`       |
| 17 | Slovak                | `slk_latn`       |
| 18 | Slovenian             | `slv_latn`       |
| 19 | Spanish (Spain)       | `spa_latn_spai`  |
| 20 | Swedish               | `swe_latn`       |

### Missing EU Languages

Excluded as not present in the original dataset:

- Danish (dan), Irish (gle), Latvian (lav), Maltese (mlt).

## Usage

Run the full Pharos EU PIQA suite:

```bash
lm_eval --model <model> --tasks pharos_global_piqa_eu
```
