# Disease Progression Risk Forecasting — Complete Guide

> **Audience:** Anyone — no prior machine learning or programming knowledge required.  
> **Project:** AgriTwin-GH | **Module:** Disease Progression & Proactive Risk Forecasting  
> **Notebook:** `notebooks/disease_progression_risk_forecasting.ipynb`

---

## Table of Contents

1. [What This Model Does (Plain English)](#1-what-this-model-does-plain-english)
2. [Why It Was Built](#2-why-it-was-built)
3. [Diseases Tracked](#3-diseases-tracked)
4. [Data Sources](#4-data-sources)
5. [How the Pipeline Works — Step by Step](#5-how-the-pipeline-works--step-by-step)
   - [Step A — Configuration & Setup](#step-a--configuration--setup)
   - [Step B — Data Loading & Validation](#step-b--data-loading--validation)
   - [Step C — Splitting Data (Train / Val / Test)](#step-c--splitting-data-train--val--test)
   - [Step D — Disease Risk Scoring](#step-d--disease-risk-scoring)
   - [Step E — Saving Early Artifacts](#step-e--saving-early-artifacts)
   - [Step F — Feature Engineering](#step-f--feature-engineering)
   - [Step G — Supervised Dataset Building](#step-g--supervised-dataset-building)
   - [Step H — Normalisation (Scaling)](#step-h--normalisation-scaling)
   - [Step I — Random Forest Model](#step-i--random-forest-model)
   - [Step J — LSTM Deep Learning Model](#step-j--lstm-deep-learning-model)
   - [Step K — High-Risk Alert Reports](#step-k--high-risk-alert-reports)
   - [Step L — Final Export & Archive](#step-l--final-export--archive)
6. [Disease Threshold Reference](#6-disease-threshold-reference)
7. [Risk Scoring Formula Explained](#7-risk-scoring-formula-explained)
8. [Feature Engineering — What Goes Into the Model](#8-feature-engineering--what-goes-into-the-model)
9. [Model Architectures](#9-model-architectures)
   - [Random Forest (RF)](#random-forest-rf)
   - [LSTM Neural Network](#lstm-neural-network)
10. [Where Everything Is Stored](#10-where-everything-is-stored)
11. [How to Read the Output](#11-how-to-read-the-output)
12. [Configuration Reference (All Tunable Parameters)](#12-configuration-reference-all-tunable-parameters)
13. [How to Run the Notebook](#13-how-to-run-the-notebook)
14. [Frequently Asked Questions](#14-frequently-asked-questions)
15. [Glossary](#15-glossary)
16. [References & Further Reading](#16-references--further-reading)

---

## 1. What This Model Does (Plain English)

Imagine you are a tomato farmer inside a greenhouse. Every hour, sensors around the greenhouse measure the **temperature, humidity, air speed, CO₂ levels, solar radiation**, and other environmental factors. Based on these readings, diseases can grow on your plants — but the damage often becomes visible only *days* after the conditions that triggered it.

This model **reads the history of those sensor readings** and **predicts, hours in advance, how likely each of five diseases is to break out** — before you can see any visible symptoms.

Specifically, it answers questions like:

> *"Given the last 24 hours of greenhouse conditions, what is the probability that Late Blight will become HIGH risk in the next 6, 12, 24, or 48 hours?"*

The output is a **risk score from 0 to 100** for each disease, mapped to a label:

| Score Range | Label | Meaning |
|-------------|-------|---------|
| 0 – 33 | 🟢 **Low** | Conditions do not favour this disease |
| 34 – 66 | 🟡 **Medium** | Borderline conditions; increase monitoring |
| 67 – 100 | 🔴 **High** | Conditions strongly favour disease outbreak |

---

## 2. Why It Was Built

Traditional disease management is **reactive** — farmers spray fungicides or pesticides after they see a problem. By then:

- The disease has already spread to many plants
- Crop losses have already occurred
- More chemical treatment is needed (higher cost, higher environmental impact)

This model is **proactive** — it gives the farmer a **6 to 48 hour early warning** so they can:

- Adjust ventilation to change humidity / air flow
- Apply targeted, timely treatment before spread
- Reduce chemical use by treating only when truly necessary
- Protect yield before symptoms appear

---

## 3. Diseases Tracked

| # | Disease | Pathogen | Key Trigger Conditions |
|---|---------|----------|----------------------|
| 1 | **Late Blight** | *Phytophthora infestans* | RH ≥ 90%, VPD ≤ 0.40 kPa, sustained ≥ 6 h, worse at night with low airflow |
| 2 | **Leaf Mold** | *Passalora fulva* | RH ≥ 90%, VPD ≤ 0.50 kPa, sustained ≥ 5 h, amplified by still air (< 1.5 m/s) |
| 3 | **Powdery Mildew** | *Oidium neolycopersici* | RH 70–90%, VPD 0.50–1.20 kPa, sustained ≥ 6 h, worse with low airflow |
| 4 | **Early Blight** | *Alternaria solani* | Temp 24–30 °C, RH ≥ 85%, sustained ≥ 4 h, amplified by high solar radiation |
| 5 | **Spider Mites** | *Tetranychus urticae* | Temp ≥ 28 °C, RH ≤ 55%, VPD ≥ 1.50 kPa, even 1 h of severe stress is risky |

> **Note:** Late Blight and Leaf Mold thrive in *wet, humid* conditions. Powdery Mildew and Spider Mites thrive in *dry, hot* conditions. Early Blight occupies the middle ground. The model tracks all five simultaneously.

---

## 4. Data Sources

### Input Files

```
data/processed/Greenhouse Indoor Conditions/
    dindigul_greenhouse_indoor_2024.csv   ← training + validation data
    dindigul_greenhouse_indoor_2025.csv   ← testing data
```

These are hourly time-series files from the **Dindigul greenhouse** in Tamil Nadu, India. They are automatically combined and validated on load.

### Sensor Columns Used

| Raw CSV Column | Internal Name | Unit | Description |
|----------------|---------------|------|-------------|
| `indoor_temp` | `temp` | °C | Air temperature inside the greenhouse |
| `indoor_humidity` | `humidity` | % | Relative humidity |
| `indoor_air_velocity` | `air_velocity` | m/s | Air movement speed (ventilation) |
| `indoor_CO2` | `co2` | ppm | Carbon dioxide concentration |
| `solarradiation` | `solar_radiation` | W/m² | Solar radiation reaching the plants |
| `day_night_flag` | `day_night_flag` | 0/1 | 0 = night, 1 = day |
| `vpd` | `vpd` | kPa | Vapour Pressure Deficit — how "dry" the air is |
| `dew_point` | `dew_point` | °C | Temperature at which air becomes saturated |

### Timeline

| Period | Dates | Role | Approximate Size |
|--------|-------|------|-----------------|
| Training | Jan 2024 → Oct 2024 | Model learns patterns | ~7,320 hours |
| Validation | Nov 2024 → Dec 2024 | Model is tuned | ~1,464 hours |
| Test | Jan 2025 → Dec 2025 | Final evaluation (unseen) | ~8,760 hours |

> ⚠️ **Chronological split — no data leakage.** The model is never allowed to peek at future data during training. It learns from the past and predicts the future, just as it would in real deployment.

---

## 5. How the Pipeline Works — Step by Step

Think of the pipeline as an assembly line. Raw sensor data goes in one end; trained models and risk predictions come out the other.

```
Raw CSV files
    │
    ▼
[A] Setup (CONFIG)
    │
    ▼
[B] Load & Validate Data
    │
    ▼
[C] Split: Train │ Val │ Test
    │
    ▼
[D] Compute Disease Risk Score (0–100) for every hour
    │
    ▼
[E] Save baseline artifacts (thresholds, splits, preview CSVs)
    │
    ▼
[F] Feature Engineering (create ~156 rich input columns)
    │
    ▼
[G] Build Training Datasets (RF tabular + LSTM windows)
    │
    ▼
[H] Normalise Features (StandardScaler — no leakage)
    │
    ├──► [I] Train Random Forest → save models + evaluate
    │
    └──► [J] Train LSTM → save model + evaluate
              │
              ▼
         [K] High-Risk Alert Report generation
              │
              ▼
         [L] Export all artifacts + unified evaluation report
```

---

### Step A — Configuration & Setup

**What happens:** All tunable parameters are defined in one Python dictionary called `CONFIG`. The notebook also automatically detects the repository root folder, resolves all file paths, and checks for required Python packages.

**Key parameters set here:**

| Parameter | Default Value | Meaning |
|-----------|---------------|---------|
| `train_end` | `2024-10-31` | Last day of training data |
| `val_end` | `2024-12-31` | Last day of validation data |
| `window_N` | `24` | How many past hours the LSTM looks at (lookback window) |
| `horizon_H` | `[6, 12, 24, 48]` | How many hours ahead to forecast |
| `seed` | `42` | Random seed (ensures every run gives the same result) |
| `mixed_precision` | `False` | Set `True` on modern NVIDIA GPUs (Ampere+) for faster training |

---

### Step B — Data Loading & Validation

**What happens:** The two CSV files (2024 and 2025) are loaded, combined into one continuous hourly time series, and inspected for quality issues.

**Checks performed:**
- All required sensor columns are present
- Missing timestamps (gaps) are detected and filled using **linear interpolation** (so if a sensor went offline for 2 hours, those values are estimated from the surrounding readings)
- Column names are renamed to standardised internal names (e.g., `indoor_temp` → `temp`)
- A gap detection report is saved listing any periods where data was missing

**Output:** A clean, gap-free DataFrame with a continuous hourly timestamp index.

---

### Step C — Splitting Data (Train / Val / Test)

**What happens:** The combined dataset is divided into three non-overlapping, **chronologically ordered** parts.

```
Jan 2024 ──────────────── Oct 2024 │ Nov─Dec 2024 │ Jan─Dec 2025
          TRAINING (learns)         VALIDATION     TEST (final score)
```

> **Why no random shuffling?** If we randomly mixed past and future data, the model could accidentally learn from the future, which would give unrealistically good results. In the real world, you only have the past — so we train on the past and test on the future.

---

### Step D — Disease Risk Scoring

**What happens:** For every hour in the dataset, the pipeline computes a **risk score from 0 to 100** for each of the five diseases. This is the "ground truth" that both models will learn to predict.

The process has four sub-steps:

**D1 — Define Thresholds:** Numeric limits for each disease (temperature ranges, humidity minimums, etc.) are defined once in `DISEASE_THRESHOLDS`. See [Section 6](#6-disease-threshold-reference) for the full table.

**D2 — Binary Condition Flag:** For each hour, a simple True/False check: *"Are conditions favourable for this disease right now?"* For example, for Late Blight: Is RH ≥ 90% AND VPD ≤ 0.40?

**D3 — Rolling Exposure Windows:** A single hour of bad conditions rarely causes disease. What matters is *how many favourable hours occurred in the last 6h, 12h, and 24h*. These rolling counts are computed for each disease.

**D4 — Risk Index Calculation:** The final risk score (0–100) is computed using a formula:

```
risk = BASE + MODIFIERS, clamped to [0, 100]
```

See [Section 7](#7-risk-scoring-formula-explained) for the full formula.

**D5/D6 — Risk Labels:** The continuous score is converted to Low / Medium / High labels.

**Output columns added per disease (example for `late_blight`):**

| Column | Type | Description |
|--------|------|-------------|
| `cond_late_blight` | 0 or 1 | Was this hour "favourable" for late blight? |
| `exposure_count_6h_late_blight` | integer | How many of the last 6 hours were favourable? |
| `exposure_count_12h_late_blight` | integer | How many of the last 12 hours were favourable? |
| `exposure_count_24h_late_blight` | integer | How many of the last 24 hours were favourable? |
| `night_exposure_24h_late_blight` | integer | Of the 24h exposure, how many occurred at night? |
| `risk_late_blight` | float 0–100 | Continuous risk score |
| `risk_label_late_blight` | string | "low" / "medium" / "high" |

---

### Step E — Saving Early Artifacts

**What happens:** Before building models, key reference files are saved so the run is reproducible and inspectable.

**Files saved:**
- `thresholds_config.json` — the exact disease threshold values used
- `data_split_summary.json` — sizes of train/val/test sets with date ranges

---

### Step F — Feature Engineering

**What happens:** Raw sensor readings alone are not enough for accurate forecasting. The model needs **context** — how was temperature trending over the last 12 hours? What was the humidity at 3 AM, not just right now?

Feature engineering transforms the 8 raw sensor columns into approximately **156 rich predictor columns**.

| Layer | What It Creates | Count |
|-------|-----------------|-------|
| **(a) Base sensors** | The 8 raw sensor values + day/night flag | 8 |
| **(b) Cyclical time** | Hour-of-day and month encoded as sine/cosine waves | 4 |
| **(c) Exposure counters** | The cond_, exposure_count_, and night_exposure_ columns from Step D | ~25 |
| **(d) Rolling statistics** | Mean, max, min, std for each sensor over 6h / 12h / 24h windows | 84 |
| **(e) Lag features** | Sensor values from 1h, 2h, 3h, 6h, 12h, 24h ago | 36 |
| **(f) Interaction terms** | temp × humidity, vpd × humidity, temperature − dew_point | 3 |
| **(g) Night-segmented rolling** | Rolling mean computed on *night hours only* for each sensor | 21 |

> **Why cyclical encoding for time?** Hour 23 and hour 0 are only 1 hour apart, but if you encode them as numbers (23 and 0), the model sees them as 23 apart. Sine/cosine encoding wraps around correctly so midnight and 11 PM are treated as neighbours.

> **Why night-only rolling stats?** Diseases like Late Blight and Leaf Mold are strongly driven by overnight humidity. Mixing day readings into the average would dilute that signal.

---

### Step G — Supervised Dataset Building

**What happens:** The enriched feature matrix is reorganised into the exact format each model needs.

**For the Random Forest (tabular format):**  
For each forecast horizon H (6h, 12h, 24h, 48h), a separate table is created:
- **Input (X):** All ~156 features at a given hour
- **Target (Y):** The disease risk scores H hours into the future
- Four separate datasets are created (one per horizon)

**For the LSTM (sliding window format):**  
The LSTM needs to see a *sequence* of past hours, not just a single snapshot.
- A window of the **last 24 hours** of all features is assembled for each position
- The corresponding target is the risk scores at each horizon H later
- This creates a 3-dimensional array: (number of samples × 24 hours × 156 features)

---

### Step H — Normalisation (Scaling)

**What happens:** Different sensors have vastly different numerical scales. Temperature might range from 20 to 35, while CO₂ might range from 400 to 1200. Without normalisation, the model would give far too much weight to CO₂ simply because it has bigger numbers.

A **StandardScaler** is applied:
- It computes the **mean and standard deviation of each column** — using **only** the training set
- All three splits (train, val, test) are then transformed using those training statistics
- This prevents the model from having any knowledge about future data distributions

> **Why fit only on training data?** If we computed statistics on the whole dataset including the future test set, the model would implicitly "know" future data properties, which is cheating. In real deployment, you only know statistics from past data.

**Artifacts saved:**
- `scaler.pkl` — the fitted scaler object (needed to normalise new incoming data at deployment)
- `feature_schema.json` — the exact ordered list of feature column names (needed to ensure inference inputs match training inputs)

---

### Step I — Random Forest Model

**What happens:** A `RandomForestRegressor` is trained for each forecast horizon (4 total). This is an **ensemble of 300 decision trees** that each independently predict risk scores, whose results are averaged.

**Why two models (RF + LSTM)?**  
Random Forests are fast, interpretable, and excellent with tabular data. They serve as a **strong baseline**. The LSTM then handles the temporal sequential dependencies that RF cannot capture.

**Configuration:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_estimators` | 300 | Number of decision trees |
| `max_depth` | 12 | Maximum depth of each tree |
| `min_samples_leaf` | 5 | Minimum samples at each leaf node |

**Output — one model file per horizon:**
```
src/agritwin_gh/models/
    dp_rf_H6_<run_id>.joblib     ← predicts risk 6 hours ahead
    dp_rf_H12_<run_id>.joblib    ← predicts risk 12 hours ahead
    dp_rf_H24_<run_id>.joblib    ← predicts risk 24 hours ahead
    dp_rf_H48_<run_id>.joblib    ← predicts risk 48 hours ahead
```

**Metrics computed:**

| Metric | What It Measures |
|--------|-----------------|
| MAE | Mean Absolute Error — average prediction error in risk points (0–100) |
| RMSE | Root Mean Square Error — penalises large errors more heavily |
| Precision (High) | Of the hours it predicted HIGH risk, what fraction were actually HIGH? |
| Recall (High) | Of the actual HIGH risk hours, what fraction did it successfully detect? |
| F1 (High) | Harmonic mean of Precision and Recall — overall high-risk detection quality |

**Feature importance plots** are generated showing which input features most influenced each prediction.

---

### Step J — LSTM Neural Network Model

**What happens:** A Long Short-Term Memory (LSTM) neural network is trained. Unlike the Random Forest, it processes the full **sequence of the last 24 hours** simultaneously, capturing how conditions evolved over time.

**Architecture (plain English):**

```
Input: last 24 hours × 156 features
    │
    ▼
LSTM Layer 1 (128 units) — learns short-term patterns
    │ → Layer Normalisation (stabilises training)
    │ → Dropout 20% (prevents memorisation)
    ▼
LSTM Layer 2 (64 units) — learns longer-term patterns
    │ → Layer Normalisation
    │ → Dropout 10%
    ▼
Shared Dense Layer (64 units, ReLU) — common representation
    │
    ├──► Output Head H6  : predicts risk 6h ahead  (5 values, one per disease)
    ├──► Output Head H12 : predicts risk 12h ahead
    ├──► Output Head H24 : predicts risk 24h ahead
    └──► Output Head H48 : predicts risk 48h ahead
```

Each output head is **clipped to [0, 100]** so predictions are always valid risk scores.

**Training safeguards:**

| Callback | Purpose |
|----------|---------|
| **EarlyStopping** (patience=15) | Stops training automatically if validation loss stops improving |
| **ModelCheckpoint** | Always saves the single best model seen during training |
| **ReduceLROnPlateau** | Halves the learning rate when progress stalls (helps escape plateaus) |
| **CSVLogger** | Records loss at every epoch for plotting |

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Maximum epochs | 150 |
| Batch size | 64 |
| Optimizer | Adam (lr = 0.001) |
| Loss function | MAE (Mean Absolute Error) |
| Early stopping patience | 15 epochs |

---

### Step K — High-Risk Alert Reports

**What happens:** A human-readable alert report is generated from the model's predictions. This is what a farmer or operator would actually receive.

**Report structure:**

```
[2025-06-15 02:00:00]  HIGH RISK ALERT

Priority 1: Late Blight — Risk: 78/100 (HIGH)
  Peak horizon: 12h ahead
  Primary driver: Sustained high humidity (≥90%) for 8h with low ventilation

Priority 2: Leaf Mold — Risk: 71/100 (HIGH)
  Peak horizon: 6h ahead
  Primary driver: VPD critically low (0.28 kPa) with overnight stagnant air

Action recommended: Increase ventilation, monitor for leaf wetness
```

Reports are aggregated using the **maximum risk across the next 24 hours** — so if H6 predicts Medium but H12 predicts High, the High alert is raised.

---

### Step L — Final Export & Archive

**What happens:** All results from RF and LSTM are compiled into a unified archive:
- A complete JSON evaluation report (all metrics for all models × horizons × diseases)
- A CSV summary table (easy to open in Excel / Google Sheets)
- 20+ sample high-risk reports from the test set
- An artifact manifest listing every file created during the run with its size

---

## 6. Disease Threshold Reference

> **Sources:** Threshold values were derived from the FAO Good Agricultural Practices guide ([PDF](General%20Research%20Papers/Good%20Agricultural%20Practices.pdf)), the UC IPM Tomato Disease Management guidelines ([ipm.ucanr.edu](https://ipm.ucanr.edu)), and the FAO corporate website ([fao.org](https://www.fao.org)). See [Section 16](#16-references--further-reading) for full citations.

### Late Blight

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `rh_min` | 90.0% | RH must be at or above this |
| `vpd_max` | 0.40 kPa | VPD must be at or below this |
| `required_hours_24h` | 6 h | Hours needed to reach base score of 100 |
| `night_boost` | +15 pts | Bonus when night exposure is high |
| `night_boost_cap` | 15 pts | Maximum bonus from this modifier |
| `low_airflow_thresh` | 0.5 m/s | Airflow below this is "low" |
| `low_airflow_boost` | +10 pts | Bonus when 24h mean airflow < 0.5 m/s |
| `low_airflow_boost_cap` | 10 pts | Maximum bonus from this modifier |

### Leaf Mold

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `rh_min` | 90.0% | RH must be at or above this |
| `vpd_max` | 0.50 kPa | VPD must be at or below this |
| `required_hours_24h` | 5 h | Hours needed to reach base score of 100 |
| `low_airflow_thresh` | 1.5 m/s | Calibrated to dataset (min AV in data = 0.76 m/s) |
| `low_airflow_boost` | +10 pts | Bonus when 24h mean airflow < 1.5 m/s |
| `night_boost` | +10 pts | Bonus for high-RH overnight periods |

### Powdery Mildew

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `rh_min` | 70.0% | RH lower bound |
| `rh_max` | 90.0% | RH upper bound |
| `vpd_min` | 0.50 kPa | VPD lower bound (unlike fungal diseases — needs *some* dryness) |
| `vpd_max` | 1.20 kPa | VPD upper bound |
| `required_hours_24h` | 6 h | Hours needed for base exposure |
| `low_airflow_boost` | +10 pts | Bonus for low ventilation |

### Early Blight

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `temp_min` | 24.0 °C | Temperature lower bound |
| `temp_max` | 30.0 °C | Temperature upper bound |
| `rh_min` | 85.0% | RH must be at or above this |
| `required_hours_24h` | 4 h | Hours needed for base exposure |
| `radiation_boost_thresh` | 200 W/m² | Solar radiation above this increases risk |
| `radiation_boost` | +12 pts | Bonus when daily mean solar > 200 W/m² |

### Spider Mites

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `temp_min` | 28.0 °C | Temperature must be at or above this |
| `rh_max` | 55.0% | RH must be at or below this (dry conditions) |
| `vpd_min` | 1.50 kPa | Air must be very dry |
| `required_hours_24h` | 1 h | Even 1 hour of severe stress is dangerous |
| `radiation_boost_thresh` | 150 W/m² | Solar radiation above this increases risk |
| `radiation_boost` | +15 pts | Bonus for intense solar radiation / heat |

---

## 7. Risk Scoring Formula Explained

The risk score for every hour is calculated in three parts:

### Part 1 — Base Score

```
BASE = 100 × min(1.0,  qualifying_hours_in_24h  /  required_hours_threshold)
```

**Example (Late Blight, required = 6 hours):**
- If 3 out of the last 24 hours had RH ≥ 90% AND VPD ≤ 0.40 → BASE = 100 × (3/6) = **50**
- If 6+ hours met the condition → BASE = **100** (saturated)

### Part 2 — Modifiers

Modifiers are **added on top of the base score** based on aggravating conditions:

**Night Boost** (Late Blight, Leaf Mold):
```
night_boost_contribution = min(cap,  boost × (night_hours_in_24h / 24))
```
If 12 of the 24 hours were at night AND met the condition → night fraction = 0.5 → contribution = 0.5 × 15 = 7.5 (capped at 15)

**Low Airflow Boost** (Late Blight, Leaf Mold, Powdery Mildew):
```
airflow_contribution = min(cap,  boost × (1 − clamp(mean_airvel / threshold, 0, 1)))
```
If mean air velocity over 24h = 0 m/s and threshold = 0.5 → ratio = 0 → contribution = boost × 1.0 = 10

**Radiation Boost** (Early Blight, Spider Mites):
```
radiation_contribution = min(cap,  boost × clamp((mean_solar − threshold) / threshold, 0, 1))
```
Linearly ramps from 0 to full boost as solar radiation goes from 200 to 400 W/m²

### Part 3 — Final Clamp

```
risk = clamp(BASE + all_modifiers,  0,  100)
```

The result can never go below 0 or above 100.

---

## 8. Feature Engineering — What Goes Into the Model

The model receives approximately **156 features** per time step. Here is a breakdown:

### (a) Base Sensors — 8 columns
The raw sensor values at the current hour: `temp`, `humidity`, `air_velocity`, `co2`, `solar_radiation`, `vpd`, `dew_point`, `day_night_flag`

### (b) Cyclical Time Encoding — 4 columns
```
hour_sin  = sin(2π × hour / 24)
hour_cos  = cos(2π × hour / 24)
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```
This tells the model "what time of day it is" and "what season it is" in a way that wraps around correctly (midnight = hour 0 ≈ hour 23).

### (c) Disease Exposure Counters — ~25 columns
The `cond_*`, `exposure_count_*`, and `night_exposure_*` columns computed in Step D. These directly encode how many favourable hours occurred recently for each disease.

### (d) Rolling Statistics — 84 columns
For each of **7 sensor variables** × **3 window sizes** (6h, 12h, 24h) × **4 statistics** (mean, max, min, std):
```
temp_mean_6h, temp_max_6h, temp_min_6h, temp_std_6h
temp_mean_12h, temp_max_12h, ...
humidity_mean_6h, ...
vpd_std_24h, ...
```

### (e) Lag Features — 36 columns
For each of **6 key variables** × **6 lag offsets** (1h, 2h, 3h, 6h, 12h, 24h):
```
temp_lag1, temp_lag6, temp_lag24
humidity_lag1, vpd_lag12, ...
```
These let the model compare current conditions to what they were hours ago (rising or falling trend).

### (f) Interaction Terms — 3 columns
```
temp_x_rh       = temperature × humidity  (heat-humidity load)
vpd_x_rh        = vpd × humidity          (stress paradox indicator)
dewpoint_spread = temperature − dew_point  (≈0 means near condensation)
```

### (g) Night-Segmented Rolling — 21 columns
For each of **7 variables** × **3 window sizes**: rolling mean computed **only over nighttime hours** (day_night_flag = 0). Daytime hours are masked out so the "night climate" signal is preserved.

---

## 9. Model Architectures

### Random Forest (RF)

**Type:** Ensemble of decision trees (scikit-learn `MultiOutputRegressor(RandomForestRegressor)`)

**How it works:** 300 decision trees each independently examine the 156 features and predict risk scores. Their predictions are averaged. Decision trees ask simple questions like *"Is humidity > 87? If yes, go left. Is VPD < 0.5? If yes, predict 72..."*

**Strengths:** Fast, interpretable, does not require normalisation, handles non-linear patterns well  
**Limitation:** Does not naturally understand temporal sequences — each row is treated independently

**Four separate models are trained** — one per forecast horizon (H = 6, 12, 24, 48 hours)

---

### LSTM Neural Network

**Type:** Long Short-Term Memory recurrent neural network (Keras/TensorFlow)

**How it works:** The LSTM processes 24 hours of data as a sequence, maintaining a "memory" that carries information from earlier hours into later computations. It learns patterns like *"after 6 consecutive high-humidity nights, risk rises sharply on the 7th"* — patterns the Random Forest cannot see.

**Architecture summary:**

```
Input: (24 hours × 156 features)
    │
LSTM Layer 1: 128 units, returns full sequence
    ↓ LayerNorm → Dropout(20%)
LSTM Layer 2: 64 units, returns final state only
    ↓ LayerNorm → Dropout(10%)
Shared Dense: 64 units (ReLU)
    │
    ├── Head H6  → Dense(32) → Dense(5) → Clip[0,100]
    ├── Head H12 → Dense(32) → Dense(5) → Clip[0,100]
    ├── Head H24 → Dense(32) → Dense(5) → Clip[0,100]
    └── Head H48 → Dense(32) → Dense(5) → Clip[0,100]
```

**One single model produces all four horizons simultaneously** via its four output heads.

**Training details:**
- Optimizer: Adam with initial learning rate 0.001
- Loss: Mean Absolute Error (MAE) across all four heads
- Early stopping: stops after 15 epochs with no improvement; restores the best checkpoint
- Learning rate scheduler: halves LR every 7 stagnant epochs, minimum 0.000001

---

## 10. Where Everything Is Stored

### Input Data
```
data/processed/Greenhouse Indoor Conditions/
    dindigul_greenhouse_indoor_2024.csv
    dindigul_greenhouse_indoor_2025.csv
```

### Processed Risk Data
```
data/processed/Disease/
    dp_<run_id>_risk_preview.csv     ← last 200 rows (quick sanity check)
    dp_<run_id>_full_risk.csv        ← complete dataset with all risk scores
```

### Trained Models
```
src/agritwin_gh/models/
    lstm_<run_id>.keras              ← LSTM neural network (single file)
    dp_rf_H6_<run_id>.joblib         ← Random Forest, predicts 6h ahead
    dp_rf_H12_<run_id>.joblib        ← Random Forest, predicts 12h ahead
    dp_rf_H24_<run_id>.joblib        ← Random Forest, predicts 24h ahead
    dp_rf_H48_<run_id>.joblib        ← Random Forest, predicts 48h ahead
```

### Run Artifacts (Everything Else)
All artifacts for a specific run are saved in a single flat folder:
```
src/agritwin_gh/models/artifacts/dp_<run_id>/
    │
    ├── thresholds_config.json           ← disease threshold parameters used
    ├── data_split_summary.json          ← train/val/test sizes and dates
    ├── scaler.pkl                       ← fitted StandardScaler (needed for inference)
    ├── feature_schema.json              ← ordered feature column list
    ├── engineered_features_head.csv     ← first 50 rows of scaled X (inspection)
    ├── engineered_features_tail.csv     ← last 50 rows of scaled X (inspection)
    │
    ├── rf_metrics.json         ← RF evaluation metrics (all diseases × horizons)
    ├── rf_metrics_summary.csv          ← same metrics as CSV (Excel-friendly)
    ├── rf_feature_importance_H6.png     ← top-30 features for H=6h RF model
    ├── rf_feature_importance_H12.png    ← top-30 features for H=12h RF model
    ├── rf_feature_importance_H24.png    ← top-30 features for H=24h RF model
    ├── rf_feature_importance_H48.png    ← top-30 features for H=48h RF model
    ├── rf_trajectory.png                ← predicted vs actual risk over 168h test window
    │
    ├── rnn_training_history.png         ← LSTM loss curve (train vs val per epoch)
    ├── rnn_trajectory_<disease>.png     ← LSTM predicted vs actual for each disease
    ├── rnn_horizon_error.png           ← how error increases with forecast distance
    ├── rnn_metrics.json        ← LSTM evaluation metrics
    ├── rnn_metrics_summary.csv         ← same metrics as CSV
    ├── rnn_training_history.csv         ← epoch-by-epoch loss log
    │
    ├── evaluation_report.json           ← UNIFIED RF + LSTM metrics report
    ├── evaluation_report_summary.csv    ← comparison table (Excel-friendly)
    ├── final_sample_reports.json        ← 20+ human-readable alert reports from test set
    └── artifact_manifest.json           ← list of ALL files with sizes
```

> **`<run_id>`** is a timestamp automatically generated when the notebook is run, e.g. `dp_20260305_111754`. This means every run is archived separately and nothing is overwritten.

---

## 11. How to Read the Output

### Evaluation Metrics (`evaluation_report_summary.csv`)

Open the file in Excel or Google Sheets. Columns:

| Column | Meaning |
|--------|---------|
| `horizon` | Which forecast horizon (H6 = 6h ahead, H48 = 48h ahead) |
| `RF_mean_MAE` | Average prediction error (in risk points 0–100) for the Random Forest |
| `LSTM_mean_MAE` | Average prediction error for the LSTM |
| `RF_F1_High` | How well the RF detects HIGH-risk events (0 = detects nothing, 1 = perfect) |
| `LSTM_F1_High` | How well the LSTM detects HIGH-risk events |

**Interpreting MAE:** An MAE of 5.0 means predictions are off by an average of 5 risk points. For a score that ranges 0–100, this is very good. An MAE of 20+ points would indicate the model is struggling.

**Interpreting F1_High:** This is the most important metric for an early warning system. Values above 0.7 are generally considered good. The 6h horizon always has better F1 than 48h because shorter-term prediction is inherently easier.

### Feature Importance Plots (`rf_feature_importance_H*.png`)

These bar charts show the top 30 most influential features for each RF model. Longer bars = more important. Common top features include:
- Rolling exposure counters (e.g., `exposure_count_24h_late_blight`)
- Recent humidity and VPD statistics (e.g., `humidity_mean_12h`, `vpd_min_6h`)
- Lag features (e.g., `humidity_lag6`, `vpd_lag12`)

### Sample Alert Reports (`final_sample_reports.json`)

Each entry in the JSON file represents one test-set moment and contains:
- `timestamp` — the hour being assessed
- `n_high` — number of diseases at HIGH risk
- `alerts` — the human-readable alert text
- `raw_risks` — exact predicted risk scores for all 5 diseases × 4 horizons

---

## 12. Configuration Reference (All Tunable Parameters)

All parameters are centralised in `CONFIG` (cell A2) and `FE_CONFIG` (cell F1). Change values there; all downstream cells read from these dictionaries.

### CONFIG — Core Parameters

```python
CONFIG = {
    # Data paths
    "csv_2024":   "data/processed/.../2024.csv",
    "csv_2025":   "data/processed/.../2025.csv",

    # Column name mapping (must match your CSV headers)
    "col_temp":     "indoor_temp",
    "col_humidity": "indoor_humidity",
    "col_airvel":   "indoor_air_velocity",
    "col_co2":      "indoor_CO2",
    "col_solar":    "solarradiation",
    "col_vpd":      "vpd",
    "col_dew":      "dew_point",

    # Split boundaries
    "train_end":  "2024-10-31",
    "val_end":    "2024-12-31",

    # Model architecture
    "window_N":   24,              # LSTM lookback hours
    "horizon_H":  [6, 12, 24, 48], # Forecast horizons

    # Random Forest
    "rf_params": {
        "n_estimators":     300,
        "max_depth":        12,
        "min_samples_leaf":  5,
    },

    # Risk label thresholds
    "risk_label_bins": {
        "low":    [0,  33],
        "medium": [34, 66],
        "high":   [67, 100],
    },
}
```

### CONFIG["lstm"] — LSTM Parameters (set in cell J0)

```python
CONFIG["lstm"] = {
    "units_1":     128,     # Layer-1 LSTM hidden units
    "units_2":     64,      # Layer-2 LSTM hidden units
    "dense_units":  64,     # Shared dense layer
    "dropout_1":   0.20,    # Dropout after LSTM-1
    "dropout_2":   0.10,    # Dropout after LSTM-2
    "layer_norm":  True,    # LayerNormalization (stabilises training)
    "lr":          1e-3,    # Initial learning rate
    "loss":        "mae",   # Loss function
    "epochs":      150,     # Maximum training epochs
    "batch_size":  64,
    "patience":    15,      # EarlyStopping patience
    "min_delta":   1e-4,
    "reduce_lr_factor":   0.50,  # LR reduction factor
    "reduce_lr_patience":  7,    # Epochs before LR reduction
    "min_lr":             1e-6,  # Minimum learning rate
}
```

---

## 13. How to Run the Notebook

### Prerequisites

- Python 3.10+  
- Virtual environment at `e:\AgriTwin-GH\.venv` (or Colab)
- Required packages: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `tqdm`  
  *(Cell A1 will automatically install missing packages)*

### Local Run (VS Code / Jupyter)

1. Open `notebooks/disease_progression_risk_forecasting.ipynb`
2. Select the kernel: `.venv` (Python 3.10+)
3. Run cells top to bottom (**Kernel → Run All**, or run each cell individually)
4. Re-running is safe — each run generates a new `run_id` and saves to a new folder

### Google Colab

1. Upload the notebook and mount Google Drive
2. In cell A2, set:
   ```python
   CONFIG["repo_root"] = "/content/drive/MyDrive/AgriTwin-GH"
   ```
3. Run all cells

### Partial Re-run (after fixing a bug)

If you need to re-run only parts:
- Changed **disease thresholds** (D1/D2): Re-run D1 → D6, then F7, G4, H3, I6, J7 onward
- Changed **feature engineering** (F cells): Re-run F7, G4, H3, I6, J7 onward
- Changed **LSTM architecture** (J2): Re-run J2 through J7 onward
- Changed **CONFIG split dates**: Re-run everything from C2 onward

### Using a Pre-trained LSTM

If you already have a trained `.keras` file:
1. Place it at: `src/agritwin_gh/models/lstm_<run_id>.keras`
2. In cell J7 (the J run cell), comment out the `train_lstm_model(...)` call and instead load:
   ```python
   lstm_model = keras.models.load_model(str(P_LSTM_MODEL_PATH))
   ```

---

## 14. Frequently Asked Questions

**Q: Why does the risk score sometimes stay at 0 for a disease?**  
A: The primary threshold condition was never met. For example, Powdery Mildew requires RH between 70–90% — if your greenhouse runs at 95% humidity all the time, its primary condition (RH ≤ 90%) is never satisfied. Check `cond_<disease>` column: if it is always 0, the threshold is not achievable in your dataset. You may need to adjust the `DISEASE_THRESHOLDS` for your specific greenhouse environment.

**Q: What does `run_id` mean and why is it in every filename?**  
A: The `run_id` is a timestamp (`dp_YYYYMMDD_HHMMSS`) automatically generated when the notebook starts. It ensures each training run is saved in its own unique folder/file, so you never accidentally overwrite a previous run. You can compare different runs by looking at their `evaluation_report.json` files.

**Q: Why are there four RF models but only one LSTM?**  
A: The Random Forest cannot naturally predict multiple time steps — it predicts one target at a time, so four separate models (one per horizon) are needed. The LSTM uses a **multi-head architecture**: a single shared encoder processes the input sequence, then four separate output "heads" simultaneously predict all four horizons in one pass.

**Q: What is the difference between `val_loss` and `test` metrics?**  
A: The model uses validation loss during training (to decide when to stop and which checkpoint is best). Test metrics are only computed after training is complete — the model has never seen the test set during learning. Test metrics are the **honest** measure of real-world performance.

**Q: What does "no data leakage" mean?**  
A: Data leakage means the model accidentally learns information from the future that would not be available in real deployment. This project guards against it in three places: (1) chronological split only, (2) StandardScaler fit on training data only, (3) rolling/lag features use only past values (`.rolling()` looks backward, `.shift()` shifts forward in time).

**Q: Can I add a new disease?**  
A: Yes. Add an entry to `DISEASE_THRESHOLDS` (cell D1) following the same structure as existing diseases. Add the disease name to `CONFIG["diseases"]`. Then add a branch in `compute_binary_condition()` (cell D2). Re-run from D1 onward.

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **Artifact** | Any file produced by the pipeline (model file, metrics JSON, plot, etc.) |
| **Batch size** | Number of training samples processed together before updating model weights |
| **Binary condition** | A True/False flag: "Is this hour favourable for this disease?" |
| **Chronological split** | Dividing the dataset by date order, not randomly |
| **Clamp** | Force a value to stay within a range, e.g. clamp(150, 0, 100) = 100 |
| **Data leakage** | When a model accidentally learns from future data it would not have in deployment |
| **Dropout** | Randomly disabling neurons during training — forces robustness, prevents memorisation |
| **Early stopping** | Automatically stopping training when the model stops improving on validation data |
| **Epoch** | One complete pass through the training dataset |
| **F1 score** | Balanced measure of Precision and Recall: 2×(P×R)/(P+R) |
| **Feature** | An input variable used by the model (e.g., `temp_mean_12h`) |
| **Feature engineering** | Creating new columns from raw data to give the model more useful information |
| **Horizon** | How many hours ahead the model predicts (e.g., H=6 means "6 hours from now") |
| **LSTM** | Long Short-Term Memory — a type of neural network that processes sequences |
| **MAE** | Mean Absolute Error — average prediction error (lower is better) |
| **ModelCheckpoint** | Keras callback that saves the model only when it improves |
| **Normalisation** | Rescaling features to have mean=0, std=1 so all features have equal weight |
| **Precision** | Of all "HIGH" predictions, what fraction were actually HIGH? |
| **Random Forest** | Ensemble of many decision trees whose predictions are averaged |
| **Recall** | Of all actual HIGH events, what fraction did the model predict as HIGH? |
| **Risk index** | The continuous score 0–100 representing disease outbreak likelihood |
| **RMSE** | Root Mean Square Error — like MAE but penalises large errors more |
| **Rolling window** | A computation that slides over the time series, e.g. "mean of the last 12 hours" |
| **Run ID** | A unique timestamp string identifying a specific training run |
| **Scaler** | Object that normalises input features during training and inference |
| **VPD** | Vapour Pressure Deficit — the difference between the amount of moisture in the air and how much it can hold; high VPD = dry air, low VPD = humid air |
| **Window N** | The LSTM lookback period: how many past hours it sees as input (default: 24h) |

---

## 16. References & Further Reading

The disease threshold parameters used in this project (temperature ranges, humidity limits, VPD bounds, required exposure hours, and all modifier values) were established by consulting the sources listed below. No threshold value was invented — each is grounded in peer-reviewed agronomic guidance or established extension recommendations.

### Primary Reference Document

| # | Document | Authors / Publisher | Access |
|---|----------|--------------------|---------|
| 1 | **Good Agricultural Practices for Greenhouse Tomato Production** | Food and Agriculture Organization of the United Nations (FAO) | [Open PDF](General%20Research%20Papers/Good%20Agricultural%20Practices.pdf) |

This document informed the **disease-favourable condition definitions** for all five pathogens, specifically:
- Humidity and temperature bands that create infection-risk windows
- The role of air circulation in fungal spore dispersal and settlement
- Recommended monitoring frequencies that shaped the hourly granularity of this model
- The conceptual basis for distinguishing *wet-disease* (Late Blight, Leaf Mold) from *dry-disease* (Powdery Mildew, Spider Mites) thresholds

---

### Online Reference Resources

| # | Resource | URL | What It Contributed |
|---|----------|-----|-----------------------|
| 2 | **UC IPM — Integrated Pest Management for Tomatoes** | [ipm.ucanr.edu](https://ipm.ucanr.edu) | Per-disease threshold values (RH%, VPD kPa, temperature °C) for Late Blight, Leaf Mold, Early Blight, Powdery Mildew, and Spider Mites; infection-period duration guidance used to set `required_hours_24h` per disease |
| 3 | **FAO — Food and Agriculture Organization of the United Nations** | [fao.org](https://www.fao.org) | Broader agronomic context for greenhouse climate management; cross-validation of disease risk scoring methodology and risk label definitions (Low / Medium / High) |

---

### Threshold-to-Source Mapping

The table below traces each disease's key threshold values to the source documents:

| Disease | Key Thresholds | Primary Source |
|---------|---------------|----------------|
| **Late Blight** | RH ≥ 90%, VPD ≤ 0.40 kPa, required 6 h / 24 h, night boost | UC IPM + FAO GAP PDF [1,2] |
| **Leaf Mold** | RH ≥ 90%, VPD ≤ 0.50 kPa, required 5 h / 24 h, airflow < 1.5 m/s | UC IPM + FAO GAP PDF [1,2] |
| **Powdery Mildew** | RH 70–90%, VPD 0.50–1.20 kPa, required 6 h / 24 h | UC IPM [2] |
| **Early Blight** | Temp 24–30 °C, RH ≥ 85%, required 4 h / 24 h, solar > 200 W/m² | UC IPM + FAO GAP PDF [1,2] |
| **Spider Mites** | Temp ≥ 28 °C, RH ≤ 55%, VPD ≥ 1.50 kPa, even 1 h triggers risk | UC IPM [2] |

> The risk scoring formula (base score + night/airflow/radiation modifiers, clamped to [0, 100]) was designed specifically for this project to quantify cumulative exposure into a single actionable index, drawing on the exposure-duration principles described in the above sources.

---

*Document generated: March 2026 | AgriTwin-GH — Precision Agriculture Digital Twin for Greenhouse Tomato Production*
