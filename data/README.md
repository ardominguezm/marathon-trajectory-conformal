# Data (not tracked)

Place these files in this folder. They are excluded by `.gitignore` because of redistribution restrictions on the
original race results. The cleaned files are those of the companion repository
[ardominguezm/marathon-km-collapse](https://github.com/ardominguezm/marathon-km-collapse).

| File | Content | Rows |
|---|---|---|
| `segment_paces_data.csv` | Boston, Chicago, New York 2021-2024, segment times (min) for 10 segments, finish time, sex, age, pooled performance quartile | 396,813 |
| `berlin_marathon_2021_2024.csv` | Raw Berlin results, cumulative split times in seconds (5K ... 40K, Finish Net) | 137,847 raw, 134,287 after cleaning |
| `wall_labels_data.csv` | Optional. Persistent-deterioration labels for the three development events, used only for diagnostic slices | 396,813 |

Cleaning follows Domínguez-Monterroza (2026, *Physical Review E*, doi:10.1103/1wch-rv2h): every segment pace in
(0, 20) min/km and finish time in (120, 480) min. Berlin is cleaned inside `mtc.data.load_berlin`.

Segment layout (km): 0-5, 5-10, 10-15, 15-20, 20-half (1.0975), half-25 (3.9025), 25-30, 30-35, 35-40, 40-finish (2.195).
A forecast "at checkpoint k" uses only the first k segments (k = 1 ... 9, i.e. 5K ... 40K).
