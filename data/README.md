# Data dictionary and publication policy

Only aggregate, non-participant-level tables are included in this public repository.

## Published data

### `aggregate/attention_by_category.csv`

One row per AOI category across the complete study.

| Column | Meaning |
|---|---|
| `category` | English AOI category name |
| `total_dwell_ms` | Total estimated dwell time in milliseconds |
| `total_gaze_count` | Number of valid gaze samples assigned to the AOI |
| `dwell_pct` | Category share of all AOI dwell time |

### `aggregate/attention_by_stimulus.csv`

One row per stimulus and AOI category, aggregated across recordings.

| Column | Meaning |
|---|---|
| `stimulus` | Landscape image label |
| `category` | English AOI category name |
| `total_dwell_ms` | Total estimated dwell time in milliseconds |
| `total_gaze_count` | Number of valid gaze samples assigned to the AOI |
| `n_recordings` | Number of recordings contributing to the row |

ANOVA output is under `results/anova`, while aggregate cluster profiles and PCA feature loadings are under `results/machine-learning`.

## Restricted data not published

The original research folder also contains participant names, IDs, gender, recording identifiers, individual attention profiles, and cluster assignments. Those files are intentionally omitted:

- participant information CSV/XLSX files;
- recording-level attention data;
- participant-by-photo composite features;
- participant-level raw and grouped AOI profiles;
- participant cluster assignments; and
- raw Tobii Studio project data and gaze coordinates.

These materials must remain in approved research storage and should only be shared under the study's consent, institutional, and data-governance requirements.
