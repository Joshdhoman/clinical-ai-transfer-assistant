# Data dictionary — `data/synthetic/transfer_center_requests.csv`

Synthetic. 9,060 rows as delivered; 9,000 after `clean()` removes 60 exact
duplicates. One row per transfer request. No real patient, facility, or payer.

| Column | Type | Nulls | Notes | Role |
|---|---|---:|---|---|
| `transfer_request_id` | string | 0 | `TR#####`; 60 collide on the duplicate rows | identifier, never a feature |
| `request_datetime` | datetime | 0 | when the request was logged | split key; source of calendar features |
| `decision_datetime` | datetime | 0 | when the transfer center decided | **post-decision — excluded** |
| `bed_assignment_datetime` | datetime | 1904 | null when not transferred | **post-decision — excluded** |
| `arrival_datetime` | datetime | 1904 | null when not transferred | **post-decision — excluded** |
| `referring_facility` | string | 0 | `Referring Facility A`–`N` | feature |
| `referring_facility_type` | string | 0 | mixed case in raw (`COMMUNITY HOSPITAL ED`); normalised in `clean()` | feature |
| `distance_miles` | float | 175 | 3–180; median-imputed in the pipeline | feature |
| `transport_mode` | string | 0 | stray whitespace in raw (`"  Ground ALS "`); trimmed in `clean()` | feature |
| `requested_service_line` | string | 0 | 9 lines (Cardiology, Trauma, …) | feature |
| `requested_level_of_care` | string | 0 | Med-Surg / Progressive Care / ICU | feature |
| `patient_age` | int | 0 | 18–99 | feature |
| `sex` | string | 0 | Male / Female | feature |
| `payer` | string | 263 | filled with `Unknown` in `clean()` | **audited, then excluded from the model** |
| `acuity_score` | float | 141 | 1 (low) – 5 (high); median-imputed | feature |
| `icu_occupancy_pct` | float | 0 | 0.55–1.00 | feature |
| `medsurg_occupancy_pct` | float | 0 | 0.57–1.00 | feature |
| `ed_boarding_count` | int | 0 | 0–28 admitted patients boarding in the ED | feature |
| `disposition` | string | 0 | `Accepted - Transferred` / `Accepted - Not Transferred` / `Declined` | **target source — excluded as a feature** |
| `decline_reason` | string | 7597 | populated only when `Declined` | **post-decision — excluded** |
| `inpatient_los_days` | float | 1904 | null when not transferred | **post-decision — excluded** |
| `icu_upgrade_within_24h` | float | 1904 | null when not transferred | **post-decision — excluded** |

## Derived in `clean()`

| Column | Definition |
|---|---|
| `declined` | `1` if `disposition == "Declined"` else `0` |
| `request_hour` | `request_datetime.hour` |
| `request_dow` | `request_datetime.dayofweek` (Monday = 0) |
| `request_month` | `request_datetime.month` |
| `is_weekend` | `1` if request falls on Sat/Sun |
| `is_overnight` | `1` if request hour ≥ 22 or < 6 |
| `split` | `train` / `validation` / `test` by request time (60 / 20 / 20) |

## Feature set

9 numeric + 2 binary + 6 categorical = 17 columns into the encoder. Full list in
`src/transfer_decline/features.py`. `FORBIDDEN` in that module is the set that
must never appear; `tests/test_data.py` asserts it.
