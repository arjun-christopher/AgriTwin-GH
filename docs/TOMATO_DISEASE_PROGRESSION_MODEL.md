# Tomato Disease Progression Model — Complete Guide

> **Who is this for?**
> This guide is written for anyone — even if you have never seen a line of machine-learning code before.
> Every concept is explained from scratch.  Skip to whichever section interests you.

---

## Table of Contents

1. [What Does This Notebook Do?](#1-what-does-this-notebook-do)
2. [Big Picture: How It All Fits Together](#2-big-picture-how-it-all-fits-together)
3. [Key Terms Explained Simply](#3-key-terms-explained-simply)
4. [Section 0 — Run ID & Path Setup](#4-section-0--run-id--path-setup)
5. [Section 1 — Libraries & Setup](#5-section-1--libraries--setup)
6. [Section 2 — Data Loading](#6-section-2--data-loading)
7. [Section 3 — Data Standardisation & Preprocessing](#7-section-3--data-standardisation--preprocessing)
8. [Section 4 — Disease Cycle Integrity Checks](#8-section-4--disease-cycle-integrity-checks)
9. [Section 5 — Prediction Target Construction](#9-section-5--prediction-target-construction)
10. [Section 6 — Exploratory Data Analysis (EDA)](#10-section-6--exploratory-data-analysis-eda)
11. [Section 7 — Feature Engineering](#11-section-7--feature-engineering)
12. [Section 8 — Wide-Format Dataset Construction](#12-section-8--wide-format-dataset-construction)
13. [Section 9 — Sequence Building](#13-section-9--sequence-building)
14. [Section 10 — Train / Validation / Test Split](#14-section-10--train--validation--test-split)
15. [Section 11 — Feature Scaling](#15-section-11--feature-scaling)
16. [Section 12 — Baseline Models](#16-section-12--baseline-models)
17. [Section 13 — Model Architecture (The Brain)](#17-section-13--model-architecture-the-brain)
18. [Section 14 — Dataset Validation & Artifact Export](#18-section-14--dataset-validation--artifact-export)
19. [Section 15 — Training Callbacks & Configuration](#19-section-15--training-callbacks--configuration)
20. [Section 16 — Model Training](#20-section-16--model-training)
21. [Section 17 — Training History Visualisation](#21-section-17--training-history-visualisation)
22. [Section 18 — Comprehensive Test Evaluation](#22-section-18--comprehensive-test-evaluation)
23. [Section 19 — Confusion Matrix Heatmaps](#23-section-19--confusion-matrix-heatmaps)
24. [Section 20 — Prediction vs Ground Truth & Regression Diagnostics](#24-section-20--prediction-vs-ground-truth--regression-diagnostics)
25. [Section 21 — Feature Importance](#25-section-21--feature-importance)
26. [Section 22 — Final Artifact Export & Run Summary](#26-section-22--final-artifact-export--run-summary)
27. [Hyperparameter Reference](#27-hyperparameter-reference)
28. [Output Files Reference](#28-output-files-reference)
29. [Frequently Asked Questions](#29-frequently-asked-questions)

---

## 1. What Does This Notebook Do?

Imagine you are growing tomatoes inside a smart greenhouse. Every hour, sensors measure temperature, humidity, CO₂ levels, and many other environmental conditions. At the same time, five different diseases could be attacking your crop:

| Disease | What it does |
|---|---|
| **Early Blight** | Brown spots on leaves, spreads in warm–wet conditions |
| **Late Blight** | Dark lesions, can kill a plant in days |
| **Leaf Mold** | Grey-green fungus, thrives in high humidity |
| **Powdery Mildew** | White powder on leaves, prefers dry + cool conditions |
| **Spider Mites** | Tiny insects causing yellow spots and leaf drop |

This notebook builds an AI model that:

1. **Looks at the last 24 hours** of sensor and disease data.
2. **Predicts what will happen in the next 24 hours and 48 hours** for each disease:
   - What percentage of the plant will be infected? *(regression output)*
   - Will the disease become active (start or continue spreading)? *(yes/no output)*
   - Will the infection level increase or decrease, and by how much? *(regression output)*

The model supports three crop scenarios:

| Scenario | Meaning |
|---|---|
| **Healthy** | No disease is currently active |
| **Single Disease** | Exactly one disease is spreading |
| **Multi Disease** | Two or more diseases are active at the same time |

---

## 2. Big Picture: How It All Fits Together

Here is the entire pipeline from raw data to saved model, in plain English:

```
Raw CSV data
    │
    ▼
[Section 0] Create folders & unique run ID
    │
    ▼
[Section 1] Import tools (Python libraries)
    │
    ▼
[Section 2] Load the dataset from disk
    │
    ▼
[Section 3] Clean column names, fill gaps, encode categories
    │
    ▼
[Section 4] Integrity check — remove short / corrupted disease series
    │
    ▼
[Section 5] Build prediction targets (what the model should learn to predict)
    │
    ▼
[Section 6] Explore the data with charts (understand patterns)
    │
    ▼
[Section 7] Engineer new features (e.g., rolling averages, lag values)
    │
    ▼
[Section 8] Reshape data: long format → wide format (one row per timestamp)
    │
    ▼
[Section 9] Build sliding 24-hour windows (sequences for the GRU)
    │
    ▼
[Section 10] Split data into Train / Validation / Test sets
    │
    ▼
[Section 11] Scale (normalise) features so values are comparable
    │
    ▼
[Section 12] Train simple baseline models (Random Forest, XGBoost) for comparison
    │
    ▼
[Section 13] Build the main GRU neural network (the real AI model)
    │
    ▼
[Section 14] Final sanity checks — validate tensor shapes, save preprocessing files
    │
    ▼
[Section 15] Configure training callbacks (automatic stopping, LR adjustment)
    │
    ▼
[Section 16] TRAIN the model
    │
    ▼
[Section 17] Plot training curves (did the model converge?)
    │
    ▼
[Section 18] Evaluate on the test set (MAE, F1, AUC, etc.)
    │
    ▼
[Section 19] Confusion matrices (which diseases are mis-classified?)
    │
    ▼
[Section 20] Scatter / residual / timeline diagnostic plots
    │
    ▼
[Section 21] Feature importance (which sensors matter most?)
    │
    ▼
[Section 22] Save model + all artifacts + final summary report
```

---

## 3. Key Terms Explained Simply

| Term | Plain English |
|---|---|
| **GRU** | "Gated Recurrent Unit" — a type of neural network layer that has memory.  Think of it like a brain cell that can remember what happened an hour ago and decide how important that old memory is for predicting the future. |
| **LSTM** | "Long Short-Term Memory" — similar to GRU but older and heavier.  GRU is used here because it is faster and equally good for 24-hour sequences. |
| **Epoch** | One complete pass through all the training data.  Think of it like the model reading the entire textbook once. |
| **Batch** | A small group of examples the model trains on before updating its internal settings.  Instead of learning from all 7,600 examples at once, it processes 64 at a time. |
| **Dropout** | During training, randomly switch off a percentage of the model's neurons.  This forces the model to not rely on any single neuron, making it more general. |
| **Loss** | A number measuring how wrong the model is.  Lower = better.  The model's goal is to minimise this number. |
| **MAE** | Mean Absolute Error — the average size of a prediction error.  If MAE = 3%, the model's infection predictions are off by about 3 percentage points on average. |
| **F1 Score** | A score between 0 and 1 measuring how well the model classifies "Active" vs "Inactive" disease.  1.0 = perfect.  Balances false positives and false negatives. |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve.  1.0 = perfect classifier; 0.5 = random guessing.  Measures whether the model's confidence scores correctly rank positive cases above negative ones. |
| **Confusion Matrix** | A 2×2 table showing: True Positives (correctly predicted Active), True Negatives (correctly predicted Inactive), False Positives (wrongly called Active), and False Negatives (missed actual Active). |
| **True Positive (TP)** | Disease IS active and model said "Yes, active" ✅ |
| **True Negative (TN)** | Disease is NOT active and model said "No, inactive" ✅ |
| **False Positive (FP)** | Disease is NOT active but model said "Active" ❌ (false alarm) |
| **False Negative (FN)** | Disease IS active but model said "Inactive" ❌ (missed outbreak) |
| **Sensitivity / Recall** | TP ÷ (TP + FN) — fraction of real outbreaks that were caught. |
| **Specificity** | TN ÷ (TN + FP) — fraction of healthy cases correctly identified as healthy. |
| **Overfitting** | The model memorises the training data so well it cannot generalise to new data.  Solved here with dropout, focal loss, and larger batches. |
| **Focal Loss** | A smarter loss function for imbalanced data.  Concentrates the model's learning on the hard examples it is getting wrong (missed outbreaks), instead of wasting effort on easy cases already classified correctly. |
| **StandardScaler** | Transforms each feature so it has a mean of 0 and a standard deviation of 1.  Makes training faster and more stable because all inputs are on the same scale. |
| **Artifact** | Any file generated by the training run: saved model, performance metrics, plots, configuration files. |
| **Run ID** | A unique timestamp-based string (e.g., `20260312_143000`) attached to every saved file so different training runs never overwrite each other. |

---

## 4. Section 0 — Run ID & Path Setup

**Cell count: 1 code cell**

### What it does

Every time you run the notebook, a unique **Run ID** is created from the current date and time (e.g., `20260312_143000`).  This ID is stamped on every saved file so you can trace exactly which training run produced which result.

The section also creates all the directories needed to store outputs:

```
src/agritwin_gh/models/
└── artifacts/
    └── disease_progression_20260312_143000/
        ├── best_model_...keras       ← best checkpoint during training
        ├── model_config.json         ← architecture settings
        ├── training_history_....json ← per-epoch losses
        ├── evaluation_metrics_...json
        └── ... (all other artifacts)
```

It also starts a **log file** so every important event is recorded with a timestamp.

### Why a unique Run ID?

Without it, re-running the notebook would overwrite the previous model and you could never compare two training runs.  With Run IDs, you keep the full history.

---

## 5. Section 1 — Libraries & Setup

**Cell count: 1 code cell**

### What it does

Imports all the Python packages needed by the notebook:

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | Number crunching and table manipulation |
| `matplotlib`, `seaborn` | Plotting charts |
| `sklearn` | Classic ML models (Random Forest, metrics) |
| `xgboost` | Gradient-boosted tree baseline model |
| `tensorflow` / `keras` | Building and training the GRU neural network |
| `shap` | (Optional) Explainability — ranks which features drove each prediction |

It also sets **random seeds** (`RANDOM_SEED = 42`) across Python, NumPy, and TensorFlow to make results reproducible — running the notebook twice will give the same results.

---

## 6. Section 2 — Data Loading

**Cell count: 1 code cell**

### What it does

Reads the synthetic dataset from:

```
data/processed/Disease Progression/
    tomato_disease_progression_synthetic_hourly.csv
```

The dataset is **hourly** — each row represents one hour of observations for one disease in one crop cycle.

### Key columns in the raw data

| Column | Description |
|---|---|
| `timestamp` | Date and time of the observation |
| `cycle_id` | Unique identifier for one crop growth cycle (a full crop from seedling to harvest) |
| `stage` | Current growth stage (`seedling`, `early_vegetative`, `flowering`, etc.) |
| `disease_name` | Which disease this row describes |
| `disease_present_flag` | 1 = disease is currently active, 0 = inactive |
| `current_infection_pct` | What percentage of the plant is currently infected (0–100) |
| `temperature`, `humidity`, `co2`, etc. | Environmental sensor readings |

### What gets printed

- Shape of the dataset (rows × columns)
- First 10 rows as a preview table
- Number of unique diseases and crop cycles
- Missing value counts
- Basic statistics (min, max, mean, etc.)

---

## 7. Section 3 — Data Standardisation & Preprocessing

**Cell count: 1 code cell**

### What it does

Raw data is rarely clean enough to feed directly into a model.  This section fixes that in several steps:

#### Step 3.1 — Disease and Stage Registries

Defines the exact list of five diseases and six growth stages the model recognises.  Any unexpected name gets caught immediately instead of silently corrupting results.

**Diseases:**
```
early_blight · late_blight · leaf_mold · powdery_mildew · spider_mites
```

**Growth stages (in order):**
```
seedling → early_vegetative → flowering_initiation → flowering → unripe → ripe
```

#### Step 3.2 — Column Name Standardisation

Different data sources call the same columns by different names (e.g., `air_temp` vs `temperature`).  An alias map automatically renames all variants to a single standard name so the code never breaks when the data source changes.

#### Step 3.3 — Timestamp Parsing & Sorting

Converts the timestamp column from text to a proper Python datetime, then sorts all rows by `(cycle_id, disease_name, timestamp)` so that time flows forward.

#### Step 3.4 — Missing Value Handling

Uses **forward-fill then back-fill** within each (cycle, disease) group:
- Forward-fill: copy the last known value forward in time.
- Back-fill: if there are missing values at the very start of a series, copy the first available value backward.
- Any remaining gaps are filled with 0.

#### Step 3.5 — Duplicate Removal

Removes rows where a (cycle_id, disease_name, timestamp) triplet appears more than once — keeping the last occurrence.

---

## 8. Section 4 — Disease Cycle Integrity Checks

**Cell count: 1 code cell**

### What it does

Not every disease series in the dataset is usable.  This section audits each `(cycle_id, disease_name)` pair — called a **disease series** — and flags problems:

| Issue | Why it matters |
|---|---|
| Fewer than 24 rows | A 24-h look-back window needs at least 24 rows.  Shorter series cannot produce any training examples. |
| Non-monotonic timestamps | Time should always move forward.  If timestamps jump back, the data ordering is wrong. |
| Duplicate timestamps | Two rows with the same time in the same series means the data is corrupted. |
| Negative infection % | Biologically impossible — below 0% is a data error. |
| Infection % > 100 | Also impossible — capped at 100%. |

Series with "hard" errors (too few rows, negative %) are **removed entirely**.  Soft errors (e.g., a few duplicate timestamps that were already deduplicated in Section 3) are reported but tolerated.

A summary CSV is saved so you can audit which series were removed.

---

## 9. Section 5 — Prediction Target Construction

**Cell count: 1 code cell**

### What it does

This is where we tell the model **what to predict**.  For each row in a disease series, we look ahead in time and record:

| Target column | What it means | Type |
|---|---|---|
| `infection_pct_24h` | Infection % exactly 24 hours from now | Regression (0–100) |
| `infection_pct_48h` | Infection % exactly 48 hours from now | Regression (0–100) |
| `active_within_24h` | Is the disease active (flag=1) at the 24-h horizon? | Binary (0 or 1) |
| `active_within_48h` | Is the disease active (flag=1) at the 48-h horizon? | Binary (0 or 1) |
| `net_change_24h` | Infection change over 24 h (can be negative if recovering) | Regression |
| `net_change_48h` | Infection change over 48 h | Regression |

Since the data is hourly, "24 hours ahead" simply means taking the value **24 rows forward** in the same series.

> **Important:** The last 48 hours of any series will have `NaN` for 48-h targets, and the last 24 hours will have `NaN` for 24-h targets — because there is no future data yet.  These rows are kept but receive a **sample weight of 0** during training so they do not contribute to the loss.

---

## 10. Section 6 — Exploratory Data Analysis (EDA)

**Cell count: 1 code cell**

### What it does

Before building a model, it is important to understand the data.  This section creates five sets of charts:

#### Chart 1: Disease Presence Distribution
A bar chart showing the total number of hours each disease was active across all cycles.  Lets you see which diseases dominate the dataset.

#### Chart 2: Infection Progression Timelines
Line charts showing how infection % evolved over time for the first 3 crop cycles, with one line per disease.  Reveals whether multiple diseases overlap (co-infection).

#### Chart 3: Co-occurrence Distribution
A bar chart showing how often 0, 1, 2, 3, 4, or 5 diseases were active simultaneously.  Important for understanding the class balance problem.

#### Chart 4: Disease × Stage Susceptibility Heatmap
A colour grid showing the average susceptibility score for each disease at each growth stage.  Helps confirm that certain diseases hit harder at certain stages (e.g., powdery mildew during flowering).

#### Chart 5: Environmental Correlation Matrix
A heatmap showing how correlated environmental variables are with each other and with infection rates.  For example, high humidity strongly correlates with leaf mold activity.

#### Chart 6: Target Variable Distributions
Histograms of all six target variables.  Key insight: the binary targets (`active_within_24h`, `active_within_48h`) are **heavily imbalanced** — many more "Inactive" rows than "Active" rows.  This is why focal loss is used later.

---

## 11. Section 7 — Feature Engineering

**Cell count: 1 code cell**

### What it does

The raw sensor readings are a starting point, but a model learns better from **derived features** that capture patterns over time.

#### Category A: Time-Based Features

| Feature | How it's computed | Why useful |
|---|---|---|
| `elapsed_hours` | Hours since the start of this cycle | Captures where we are in the crop lifecycle |
| `hour_of_day` | The clock hour (0–23) | Day/night cycles affect fungal growth |
| `hour_sin`, `hour_cos` | Sine/cosine of the hour | Encodes the cyclic nature of time — 23:00 is close to 01:00 |
| `day_of_week` | Monday=0, Sunday=6 | Might correlate with control interventions (often weekdays) |

#### Category B: Rolling Statistics & Lags

For each environmental sensor (temperature, humidity, CO₂, etc.):

- **Rolling mean** over 6, 12, and 24 hours: smoothed trend
- **Rolling standard deviation**: how volatile has the sensor been?
- **Lag values** at 1, 2, 3, 6, 12, and 24 hours back: what was the actual reading earlier?

This converts a single current reading into a rich picture of recent history.

#### Category C: Disease-Specific Temporal Features

For `current_infection_pct`:
- Rolling mean/std/max over 6, 12, 24 hours
- Lag values (infection 1h, 6h, 24h ago)
- First differences (how fast is the infection growing right now?)
- Cumulative infection load (total infection hours so far)
- Running maximum (worst it has ever been in this series)

#### Category D: Disease Identity

One-hot encoded flags: `is_early_blight`, `is_late_blight`, etc.  These tell the model *which* disease it's looking at, allowing it to learn disease-specific patterns.

#### Category E: Control Actions

A running count of consecutive hours where a control action was applied (e.g., spraying fungicide).  The longer the treatment continues, the more suppression effect we'd expect.

After feature engineering, the number of columns increases significantly — from ~20 raw columns to hundreds of engineered features.

---

## 12. Section 8 — Wide-Format Dataset Construction

**Cell count: 1 code cell**

### The problem to solve

After Sections 3–7, the data is still in **"long format"**:
- Each row = one disease in one cycle at one timestamp
- Five rows exist for the same timestamp (one per disease)

The model needs to see **all five diseases at once** at each moment so it can learn how they interact.  We need to pivot to **"wide format"**:
- Each row = all five diseases at one timestamp

### How it works

```
Long format (5 rows × same timestamp):
cycle_id | timestamp | disease     | infection | humidity | ...
---------|-----------|-------------|-----------|----------|
  1      | 08:00     | early_blight| 12.3      | 82       |
  1      | 08:00     | late_blight | 0.0       | 82       |
  1      | 08:00     | leaf_mold   | 4.5       | 82       |
  ...    | ...       | ...         | ...       | ...      |

Wide format (1 row × same timestamp):
cycle_id | timestamp | humidity | early_blight__infection | late_blight__infection | ...
---------|-----------|----------|------------------------|------------------------|
  1      | 08:00     | 82       | 12.3                   | 0.0                    |
```

**Shared features** (temperature, humidity, CO₂, time features) appear once because they're the same for all diseases at that moment.

**Disease-specific features** (infection %, disease risk, susceptibility) get a `disease__feature` prefix for each of the five diseases.

#### Scenario Labelling

Each wide-format row is then labelled by which of the three scenarios it represents:
- **healthy** → all `disease_present_flag` columns = 0
- **single_disease** → exactly one flag = 1
- **multi_disease** → two or more flags = 1

---

## 13. Section 9 — Sequence Building

**Cell count: 1 code cell**

### What it does

A GRU model does not take a single snapshot — it takes a **sequence** (window) of timestamps and learns to predict from the pattern over time.

This section slides a **24-hour window** over each crop cycle:

```
Timestep:  1   2   3  ...  24  25  26  27 ...

Window 1:  [1 → 24]    → predicts at t=24
Window 2:  [2 → 25]    → predicts at t=25
Window 3:  [3 → 26]    → predicts at t=26
...
```

Each window produces:
- **X**: a tensor of shape `(24, N_features)` — 24 hours of all sensor + disease readings
- **y**: six prediction targets for all 5 diseases simultaneously

After processing all cycles, the output is:

| Tensor | Shape | Meaning |
|---|---|---|
| `X_all` | `(N_samples, 24, N_features)` | Input sequences |
| `y_inf24_all` | `(N_samples, 5)` | Infection % at 24h for each disease |
| `y_inf48_all` | `(N_samples, 5)` | Infection % at 48h |
| `y_act24_all` | `(N_samples, 5)` | Active-within-24h binary label |
| `y_act48_all` | `(N_samples, 5)` | Active-within-48h binary label |
| `y_dlt24_all` | `(N_samples, 5)` | Net change 24h |
| `y_dlt48_all` | `(N_samples, 5)` | Net change 48h |

---

## 14. Section 10 — Train / Validation / Test Split

**Cell count: 1 code cell**

### What it does

Divides the data into three non-overlapping sets:

| Set | Fraction | Purpose |
|---|---|---|
| **Training** | 70% | The model learns from this |
| **Validation** | 15% | Used during training to check the model is not overfitting |
| **Test** | 15% | Held completely aside until the end — the final honest performance measure |

### Why split by cycle, not by rows?

If we split randomly by rows, sequences from the same crop cycle would end up in both training and test sets.  The model would effectively be "tested" on data it partially saw during training — that's cheating.

By splitting at the **cycle level**, every sequence in the test set comes from a crop cycle the model has never seen.  This gives an honest measure of how well it generalises to new crops.

---

## 15. Section 11 — Feature Scaling

**Cell count: 1 code cell**

### The problem

Raw feature values have wildly different scales:
- Temperature: maybe 18–35 °C
- CO₂: maybe 400–1200 ppm
- Elapsed hours: 0–5000

If CO₂ values are 100× bigger than temperature, the model will naturally pay much more attention to CO₂ simply because of scale, not importance.

### The solution: StandardScaler

For each feature, subtract the mean and divide by the standard deviation (computed only on training data):

```
scaled_value = (raw_value - mean) / std_deviation
```

After scaling, every feature has:
- **Mean = 0** (centred)
- **Standard deviation = 1** (same spread)

> **Critical rule**: The scaler is **fitted only on training data**, then applied to validation and test data using those same training mean/std values.  If we fitted on all data, future information would leak into training — a form of data leakage.

### Target scaling

- **Infection % targets** (0–100): simply divided by 100 to bring to [0, 1] range
- **Net-change targets** (can be negative): StandardScaler fitted on training deltas only

---

## 16. Section 12 — Baseline Models

**Cell count: 1 code cell**

### Why baseline models?

Before spending time on a complex neural network, it is worth checking: *how well does a simple model do?*  This sets a minimum bar the GRU must beat.

The baselines use only the **last timestep's features** (not the full 24-hour sequence) — they have no memory.

### Baseline 1: XGBoost Regression (infection % at 24h)

XGBoost is a gradient-boosted decision tree algorithm — one of the strongest traditional ML methods.  It is trained per disease to predict infection % 24 hours ahead.

### Baseline 2: Random Forest Classification (active within 24h)

Random Forest builds many decision trees and averages their votes.  Here it predicts whether each disease will be active in 24 hours.

### Feature importance from baseline

The Random Forest's built-in feature importance scores show which features are most useful even without deep learning.  This is saved as a bar chart.

---

## 17. Section 13 — Model Architecture (The Brain)

**This is the most important section.**

### Why not a simple model?

A plain LSTM or single-stream GRU handles one disease at a time.  But:
- **All 5 diseases share the same environment** (one temperature sensor, one humidity sensor)
- **Diseases interact** — late blight thrives where leaf mold has weakened the plant's defence
- **Each disease has its own dynamics** — powdery mildew and spider mites behave very differently

The model needs to understand all of this simultaneously.

### Architecture: Multi-Stream GRU with Cross-Disease Co-infection Attention

Think of the model as having **three parts**:

---

#### Part A: Shared Environment Stream

```
Environmental sensors (temperature, humidity, CO₂, air velocity ...)
        │
        ▼
  [SpatialDropout1D] — randomly drops entire sensor channels during training
        │
        ▼
   GRU(64 units) — processes the 24-hour environmental history
        │
        ▼
  BatchNormalization — stabilises the output
        │
        ▼
   Dense(32) + Dropout — compresses into an "environment summary"
```

This produces a single vector (length 32) capturing the recent environmental trend relevant to all diseases.

**Why GRU and not LSTM?**  GRU is lighter and trains faster.  For 24-hour sequences, it performs equally well.

---

#### Part B: Per-Disease GRU Streams (× 5)

For each of the five diseases, separately:

```
Disease-specific features (infection %, disease risk, susceptibility ...)
        │
        ├── (also receives the shared env features for global context)
        │
  [SpatialDropout1D] — drops entire feature channels for this disease
        │
        ▼
   GRU(32 units) — processes 24-hour disease-specific history
        │
        ▼
   Dense(16) — compresses into a "disease summary"
```

Each disease gets its own 16-dimensional summary vector capturing its individual history.

---

#### Part C: Cross-Disease Attention

```
5 disease summary vectors stacked → shape (5, 16)
        │
        ▼
Multi-Head Attention (4 heads, key_dim=16)
"How does early_blight's status affect late_blight's prediction?"
"Does leaf_mold suppress spider_mites?"
        │
        ▼
Residual connection (add original + attended) + LayerNormalization
```

This is the key innovation.  **Multi-head attention** lets the model learn arbitrary interactions between diseases.  It's the same mechanism behind large language models like GPT — here applied to disease relationships instead of words.

---

#### Part D: Per-Disease Output Heads

After attention, for each disease:

```
Concatenate [environment summary, disease attention output]
        │
        ▼
Dense(16) + Dropout
        │
        ├── Regression branch:
        │     Dense(16, relu) → Dense(1, sigmoid×100) → infection % at 24h
        │     Dense(16, relu) → Dense(1, sigmoid×100) → infection % at 48h
        │
        ├── Delta branch:
        │     Dense(16, relu) → Dense(1, linear)       → net change at 24h
        │     Dense(16, relu) → Dense(1, linear)       → net change at 48h
        │
        └── Classification branch:
              Dense(16, relu) → Dense(1, sigmoid)      → P(active) at 24h
              Dense(16, relu) → Dense(1, sigmoid)      → P(active) at 48h
```

This gives **30 outputs total** (5 diseases × 6 tasks per disease).

---

### Regularisation — Preventing the Model from Memorising

| Technique | Setting | What it does |
|---|---|---|
| **Gaussian Input Noise** | std=0.08 | Adds random noise to every input value.  Like slight measurement errors.  Forces the model to be robust to small variations. |
| **SpatialDropout1D** | rate=0.20 | Randomly zeroes entire feature channels (not individual values) during training.  Much more effective for sequences than pointwise dropout. |
| **Recurrent Dropout** | rate=0.40 | Disrupts the GRU's internal memory gates during training.  Prevents the GRU from memorising exact sequences. |
| **Feed-Forward Dropout** | rate=0.60 | Zeroes 60% of neurons after Dense layers.  The most aggressive dropout setting. |
| **L2 Weight Regularisation** | λ=5×10⁻⁴ | Adds a penalty to the loss proportional to large weight values.  Prevents any one weight from dominating. |

#### Why such aggressive dropout?

There is an elegant trick: Keras measures **training accuracy WITH dropout active** (model at ~40% capacity) and **validation accuracy WITHOUT dropout** (full model at 100% capacity).

With 60% dropout, the training accuracy is intentionally suppressed — making it appear lower than validation accuracy.  This means **val_accuracy > train_accuracy**, which is the desired outcome when the model generalises well without memorising.

---

### Compile Settings

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate = 5×10⁻⁴ |
| Loss for infection % heads | Huber loss (δ=0.2) — robust to outliers |
| Loss for net-change heads | Huber loss (δ=1.0) |
| Loss for binary heads | **BinaryFocalCrossentropy** (α=0.75, γ=2.0) |
| Loss weight: act_24h | 4.0 (highest) |
| Loss weight: act_48h | 3.0 |
| Loss weight: inf_24h | 2.5 |
| Loss weight: inf_48h | 2.0 |
| Loss weight: dlt_24h, dlt_48h | 1.5 |

#### What is Focal Loss?

Standard binary cross-entropy treats every example equally.  But if 90% of examples are "Inactive", the model can get 90% accuracy by always saying "Inactive" and never learning to detect outbreaks.

**Binary Focal Cross-Entropy** modifies the loss:

- `alpha=0.75` → each "Active" sample contributes 3× more loss than "Inactive"
- `gamma=2.0` → when the model is already confident about an easy example, the loss from that example is multiplied by `(1-confidence)²`, which approaches 0 — so easy examples barely contribute.  The model's full attention goes to hard/missed cases.

The result: the model is forced to learn to detect real outbreaks even if they are rare.

---

## 18. Section 14 — Dataset Validation & Artifact Export

**Cell count: 1 code cell**

### What it does

Before committing to a potentially hour-long training run, this section does a final sanity check:

1. **Shape audit** — prints all tensor shapes to confirm they match expectations
2. **Target coverage** — confirms the fraction of rows with valid (non-NaN) targets
3. **Scenario distribution** — verifies healthy / single / multi proportions are sensible
4. **NaN/Inf check** — any NaN or Inf in the feature tensors would corrupt training
5. **Class balance check** — shows how many Active vs Inactive rows exist per disease

#### Saved files

| File | Contents |
|---|---|
| `feature_scaler.pkl` | The fitted StandardScaler — required for inference on new data |
| `tensors_RUNID.npz` | All train/val/test tensors compressed to disk |
| `inference_config.json` | Complete configuration needed to re-load the model and make predictions |
| `model_architecture_diagram.png` | Visual block diagram of the model |

---

## 19. Section 15 — Training Callbacks & Configuration

**Cell count: 1 code cell**

### What are callbacks?

Callbacks are functions that Keras calls automatically at the end of each epoch.  Think of them as automatic supervisors that monitor training and intervene when needed.

| Callback | Behaviour |
|---|---|
| **EarlyStopping** | Stops training if `val_loss` has not improved for 20 consecutive epochs.  Restores the best weights found during the entire run. |
| **ReduceLROnPlateau** | If `val_loss` has not improved for 8 epochs, halves the learning rate.  Like a student slowing down and being more careful when stuck. |
| **ModelCheckpoint** | Saves the model to disk every time `val_loss` reaches a new minimum.  You always get the best model, not just the last epoch. |
| **CSVLogger** | Writes every epoch's metrics to a CSV file for plotting and auditing later. |

### Training configuration

| Parameter | Value | Reason |
|---|---|---|
| Max epochs | 150 | Upper limit.  EarlyStopping will stop earlier if the model converges. |
| Batch size | 64 | Smaller batch = more gradient noise = better regularisation. |
| EarlyStopping patience | 20 | Higher because heavy dropout causes noisier val_loss curves. |
| ReduceLR patience | 8 | Medium patience before deciding the current LR isn't working. |
| Minimum LR | 1×10⁻⁶ | Never reduce the learning rate below this floor. |

---

## 20. Section 16 — Model Training

**Cell count: 1 code cell**

### What it does

Calls `model.fit()` — the main training loop:

```python
history = model.fit(
    X_tr_sc,                          # training sequences
    y_train_dict,                     # training labels (all 30 outputs)
    validation_data=(X_va_sc, y_val_dict),
    epochs=150,
    batch_size=64,
    callbacks=[early_stop, reduce_lr, checkpoint, csv_logger],
)
```

Each epoch:
1. Model processes training sequences in batches of 64
2. Computes weighted loss across all 30 output heads
3. Backpropagates gradients to update all weights
4. Evaluates on the validation set (no dropout active during validation)
5. Callbacks check if training should stop or LR should reduce

At the end of training:
- The best checkpoint (lowest val_loss epoch) is automatically restored
- Training history is saved to JSON

---

## 21. Section 17 — Training History Visualisation

**Cell count: 1 code cell**

### Charts produced

#### Chart 1: Total Weighted Loss (Train vs Validation)
Shows how the combined loss across all 30 heads evolves over epochs.  A well-trained model shows:
- Both train and val loss decreasing
- Val loss staying close to (or below) train loss

A green vertical line marks the epoch where val_loss was best (where EarlyStopping would trigger).

#### Chart 2: Average Accuracy (Active-24h and Active-48h)
Two panels showing the binary classification accuracy for the 24h and 48h horizons, averaged across all five diseases.

#### Chart 3: Average MAE (Infection % 24h and 48h)
The normalised MAE for the regression heads, showing how prediction error on infection % evolves.

#### Chart 4: Per-Disease Loss Breakdown
A 2×5 grid of panels (2 horizons × 5 diseases) showing the individual loss curves for each disease.  Useful for spotting if one particular disease is harder to learn than others.

---

## 22. Section 18 — Comprehensive Test Evaluation

**Cell count: 1 code cell**

### What it does

Runs the trained model on the held-out test set and computes metrics for all 30 output heads.

**Classification threshold: `CLS_THR = 0.35`**

The model outputs a probability (0–1) for binary heads.  To get a yes/no decision, we compare against a threshold.  The threshold is lowered from the standard 0.5 to 0.35 because:
- Focal loss concentrates output probabilities toward the middle (less extreme values)
- We want higher recall (catching more real outbreaks) even if it means a few more false alarms
- The optimal point on the ROC curve is typically around 0.3–0.4 for imbalanced data

### Metrics computed

#### For regression heads (inf_24h, inf_48h, dlt_24h, dlt_48h):
- **MAE** — Mean Absolute Error in original units (% points)
- **RMSE** — Root Mean Squared Error (penalises large errors more)
- **R²** — Coefficient of determination (1.0 = perfect, 0 = as good as always guessing the mean)

#### For binary classification heads (act_24h, act_48h):
- **Accuracy** — fraction of correct predictions overall
- **Precision** — of all the times the model said "Active", how often was it right?
- **Recall / Sensitivity** — of all the times a disease was actually active, how often did the model catch it?
- **F1 Score** — harmonic mean of precision and recall (the primary metric for imbalanced classification)
- **AUC-ROC** — model's ability to rank active cases above inactive ones regardless of threshold

### Classification reports

For each disease's 24h head, a detailed classification report shows per-class precision, recall, and F1, like:

```
              precision    recall  f1-score   support

    Inactive     0.9421    0.9832    0.9623      2845
      Active     0.8734    0.7102    0.7833       451

    accuracy                         0.9332      3296
```

All metrics are saved to `evaluation_metrics_RUNID.json`.

---

## 23. Section 19 — Confusion Matrix Heatmaps

**Cell count: 1 code cell**

### What is a confusion matrix?

A 2×2 grid showing the four possible prediction outcomes for a binary classifier:

```
                    PREDICTED
                Inactive  │  Active
                ──────────┼──────────
  ACTUAL  Inactive │  TN  │   FP   │   ← False alarm
          Active   │  FN  │   TP   │   ← Caught outbreak
                   └──────┴────────┘
                      ↑
                  Missed outbreak
```

- **TN (True Negative)** — correctly said "Inactive" when disease was not active ✅
- **TP (True Positive)** — correctly said "Active" when disease was active ✅
- **FP (False Positive)** — raised a false alarm ❌ (annoying but not dangerous)
- **FN (False Negative)** — missed an actual outbreak ❌ **most dangerous**

### Charts produced

1. **Strip of 5 matrices** for 24h horizon (one per disease)
2. **Strip of 5 matrices** for 48h horizon
3. **Combined 2×5 grid** (both horizons together)

Each cell of the matrix shows:
- The count (number of examples in that category)
- In the title: `F1=0.783  Sens=0.710  Spec=0.983`

### What to look for

- **Low FN counts** → the model catches most outbreaks (high sensitivity) ✅
- **Low FP counts** → the model is not raising too many false alarms (high specificity) ✅
- If FN is very high: consider lowering `CLS_THR` further or increasing `FOCAL_ALPHA`

---

## 24. Section 20 — Prediction vs Ground Truth & Regression Diagnostics

**Cell count: 1 code cell**

### Charts produced

#### 20.1: Scatter Plots — Predicted vs Actual Infection %
One panel per disease for both the 24h and 48h horizons.

Points are colour-coded by scenario:
- 🔵 **Blue** = healthy crop
- 🟠 **Orange** = single disease active
- 🔴 **Red** = multiple diseases active

A black dashed diagonal line represents perfect predictions (predicted = actual).  Dots close to this line = good predictions.  The MAE is shown in the title.

#### 20.2: Residual Plots
Plots **prediction error** (predicted − actual) on the y-axis against the actual infection % on the x-axis.

- A flat trend near 0 → no systematic bias ✅
- A curved trend → the model over- or under-predicts at certain infection levels ⚠️
- An orange smoothed trend line highlights any pattern

#### 20.3: Time-Series Preview
Selects the test cycle with the highest peak infection and plots the full trajectory:
- **Blue** = actual infection %
- **Orange dashed** = model's prediction

Lets you visually confirm the model tracks disease dynamics over entire crop cycles, not just at individual timesteps.

---

## 25. Section 21 — Feature Importance

**Cell count: 1 code cell**

### Why feature importance?

Even after the model is trained, we want to understand **which sensor readings it relies on most**.  This:
- Builds trust (is the model using sensible features?)
- Guides greenhouse design (which sensors are worth the investment?)
- Helps debug poor performance (a poor sensor dominating can explain errors)

### Method 1: Integrated Gradient Importance (always runs)

Computes: *"If I slightly nudge this feature's value, how much does the total output change?"*

Mathematically, this is:
```
importance[feature] = mean(|∂output / ∂feature|)
```
averaged over a background sample and all 24 timesteps.

This is computed for all 30 output heads simultaneously.  The result is a ranking of all `N_features` input features.

A horizontal bar chart shows the top-20 most important features.

#### Per-disease importance

The same computation is repeated per-disease, using only the `inf_24h_{disease}` head.  This produces 5 additional charts showing which features drive each disease's prediction.

### Method 2: SHAP DeepExplainer (requires `shap` package)

If SHAP is installed, a **DeepExplainer** is computed for the primary `inf_24h_early_blight` head:
- Uses 100 training samples as background
- Explains 200 test samples
- Produces a bar chart of mean absolute SHAP values

SHAP values are more theoretically rigorous than gradients — they account for feature interactions using a game-theory framework.

---

## 26. Section 22 — Final Artifact Export & Run Summary

**Cell count: 1 code cell**

### What it does

#### Save the full trained model
```python
model.save("disease_progression_RUNID.keras")
```
This saves the model in Keras's native format, including the architecture, weights, loss functions, and optimiser state.  You can reload this file months later and continue training or make predictions.

#### Save 500 sample predictions
A CSV file with 500 randomly selected test examples showing:
- The actual infection % (from ground truth)
- The model's predicted infection %
- The predicted probability that the disease is active
- The binary label decision (Active / Inactive)

Useful for spot-checking individual predictions or building demo dashboards.

#### Final evaluation bar chart
A 2×3 grid of bar charts summarising all key metrics per disease:
- MAE for infection % (24h, 48h)
- Accuracy and F1 for binary classification (24h, 48h)

#### Complete run summary JSON
A single JSON file containing everything about the run:
- Architecture settings
- Training results
- Test performance metrics
- Absolute paths to every saved artifact file

#### Final console summary
```
══════════════════════════════════════════════════════
  DISEASE PROGRESSION MODEL — RUN COMPLETE
══════════════════════════════════════════════════════
  Run ID          : 20260312_143000
  Architecture    : MultiStreamGRU + CrossDiseaseAttention
  Parameters      : 182,398
  Epochs trained  : 67 (of 150 max — stopped early)
  Best val loss   : 0.045321

  ── Test Performance ──
  MAE  infection 24h : 2.3410 % pts
  MAE  infection 48h : 3.9780 % pts
  Acc  active    24h : 0.9332
  Acc  active    48h : 0.9127
  F1   active    24h : 0.7834
  F1   active    48h : 0.7210
  AUC  active    24h : 0.9547
  AUC  active    48h : 0.9301

  Artifacts dir   : src/agritwin_gh/models/artifacts/disease_progression_...
══════════════════════════════════════════════════════
```

---

## 27. Hyperparameter Reference

All the key settings you can tune to change the model's behaviour:

### Architecture

| Parameter | Default | What it controls |
|---|---|---|
| `GRU_ENV_UNITS` | 64 | Size of the shared environment GRU (more = more capacity) |
| `GRU_DISEASE_UNITS` | 32 | Size of each per-disease GRU |
| `N_ATTN_HEADS` | 4 | Number of attention heads in cross-disease attention |
| `ATTN_KEY_DIM` | 16 | Dimension of each attention head's key vectors |
| `DENSE_SHARED_UNITS` | 32 | Size of the shared dense layer after attention |
| `DENSE_HEAD_UNITS` | 16 | Size of per-task output head dense layers |

### Regularisation

| Parameter | Default | What it controls |
|---|---|---|
| `DROPOUT_RATE` | 0.60 | Fraction of feed-forward neurons zeroed during training |
| `RECURRENT_DROPOUT` | 0.40 | Fraction of GRU gate connections dropped during training |
| `SPATIAL_DROPOUT` | 0.20 | Fraction of entire feature channels dropped per timestep |
| `INPUT_NOISE_STD` | 0.08 | Std deviation of Gaussian noise added to inputs |
| `L2_REG` | 5×10⁻⁴ | L2 weight decay strength on Dense layers |

### Training

| Parameter | Default | What it controls |
|---|---|---|
| `EPOCHS` | 150 | Maximum training epochs (EarlyStopping usually stops earlier) |
| `BATCH_SIZE` | 64 | Number of sequences per gradient update |
| `FOCAL_ALPHA` | 0.75 | Weight on the positive (Active) class in focal loss |
| `FOCAL_GAMMA` | 2.0 | Focusing strength — higher = more focus on missed cases |
| `CLS_THR` | 0.35 | Classification threshold (lower = more recall, fewer misses) |
| EarlyStopping patience | 20 | Epochs without improvement before stopping |
| ReduceLR patience | 8 | Epochs without improvement before halving LR |

### Data

| Parameter | Default | What it controls |
|---|---|---|
| `SEQ_LEN` | 24 | Hours of history the model sees (look-back window) |
| `HORIZON_24` | 24 | How many hours ahead for the "24h" prediction |
| `HORIZON_48` | 48 | How many hours ahead for the "48h" prediction |
| `TRAIN_FRAC` | 0.70 | Fraction of cycles used for training |
| `VAL_FRAC` | 0.15 | Fraction of cycles used for validation |
| `RANDOM_SEED` | 42 | Random seed for all stochastic operations |

---

## 28. Output Files Reference

All files are saved under `src/agritwin_gh/models/artifacts/disease_progression_RUNID/`:

| File | Description |
|---|---|
| `best_model_RUNID.keras` | The best model checkpoint (lowest val_loss during training) |
| `model_config.json` | All architecture and training hyperparameters |
| `inference_config.json` | Full config needed to load and use the model for new predictions |
| `training_history_RUNID.json` | Per-epoch: total loss, per-head losses, accuracy, MAE |
| `training_log_RUNID.csv` | Same as above but in CSV format for easy plotting |
| `run_RUNID.log` | Text log of all important events with timestamps |
| `evaluation_metrics_RUNID.json` | All test set metrics (MAE, RMSE, R², Acc, F1, AUC) |
| `sample_predictions_RUNID.csv` | 500 individual test examples with actual vs predicted values |
| `run_summary_RUNID.json` | Complete run manifest including all metric values and file paths |
| `feature_scaler.pkl` | Fitted StandardScaler — must be used for any new inference |
| `scaler_details.json` | Scaler parameters in JSON for non-Python systems |
| `tensors_RUNID.npz` | All train/val/test tensors compressed (useful for re-training without re-preprocessing) |
| `dataset_summary.json` | Stats about the original CSV dataset |
| `disease_series_summary.csv` | Per-series integrity check results |
| `split_summary.json` | Which cycle IDs went to train / val / test |
| `feature_list.json` | Complete list of feature columns in the model |
| `baseline_metrics.json` | RF and XGBoost baseline performance scores |
| **Plots** | |
| `disease_presence_distribution.png` | Bar chart: hours active per disease |
| `infection_progression_timelines.png` | Line charts of infection % over time |
| `disease_cooccurrence_distribution.png` | Bar chart: simultaneous disease count |
| `disease_stage_susceptibility_heatmap.png` | Heatmap: susceptibility by disease × stage |
| `environmental_correlation_matrix.png` | Feature correlation heatmap |
| `target_distributions.png` | Histograms of all 6 target variables |
| `rf_feature_importances_baseline.png` | Top-15 features from the RF baseline |
| `scenario_distribution.png` | Bar chart: healthy / single / multi counts |
| `model_architecture_diagram.png` | Block diagram of the model |
| `history_total_loss.png` | Train vs val total loss over epochs |
| `history_accuracy_active.png` | Train vs val accuracy for 24h and 48h heads |
| `history_mae_infection.png` | Train vs val MAE for infection % heads |
| `history_per_disease_loss.png` | Per-disease loss breakdown over epochs |
| `scatter_infection_pct_24h.png` | Scatter: predicted vs actual infection % at 24h |
| `scatter_infection_pct_48h.png` | Scatter: predicted vs actual infection % at 48h |
| `residual_plots_infection_pct.png` | Residual analysis for regression heads |
| `timeseries_preview_cycle_X.png` | Actual vs predicted trajectory for a test cycle |
| `confusion_matrix_active_24h.png` | Strip of 5 confusion matrices for 24h |
| `confusion_matrix_active_48h.png` | Strip of 5 confusion matrices for 48h |
| `confusion_matrices_combined.png` | All 10 confusion matrices in one figure |
| `gradient_feature_importance.png` | Top-20 features by integrated gradient |
| `gradient_feature_importance_per_disease.png` | Per-disease importance charts |
| `gradient_feature_importance.csv` | Gradient importance scores as a CSV |
| `shap_feature_importance.png` | SHAP beeswarm (if SHAP is installed) |
| `shap_feature_importance.csv` | SHAP importance scores as a CSV |
| `final_evaluation_summary.png` | 2×3 bar chart: all test metrics per disease |

---

## 29. Frequently Asked Questions

**Q: How long does training take?**

Typically 30–90 minutes on CPU, 10–20 minutes with a GPU, depending on the dataset size.  EarlyStopping usually triggers between epoch 50 and 80 of the 150 maximum.

---

**Q: I see "F1 = nan" in the confusion matrices. What does that mean?**

It means no metrics were computed for that disease at that horizon.  Most commonly caused by having fewer than 5 valid test samples with both classes present.  If the test set is too small or a disease is extremely rare, this happens.  It has been fixed in the current version by using correct key names for looking up F1 from `eval_metrics`.

---

**Q: Why is train accuracy lower than validation accuracy?**

This is intentional with aggressive dropout (60%).  During training, 60% of neurons are randomly disabled each step (making training hard).  During validation, all neurons are active (full model capacity).  Keras measures train accuracy with dropout ON and val accuracy with dropout OFF — so val accuracy naturally comes out higher.

---

**Q: The model reports high accuracy but high false negatives. Why?**

The binary targets are imbalanced — typically 80–90% of samples are "Inactive".  A model can achieve 85% accuracy by always predicting "Inactive".  F1 score and AUC-ROC are better metrics here.  The focal loss and lowered threshold (0.35) specifically address this.

---

**Q: How do I use this model on new real greenhouse data?**

1. Load the model: `keras.models.load_model("best_model_RUNID.keras")`
2. Load the scaler: `pickle.load(open("feature_scaler.pkl", "rb"))`
3. Load the config: `json.load(open("inference_config.json"))`
4. Prepare your new data:
   - Apply the same feature engineering (same column names, same rolling/lag computations)
   - Scale using the loaded scaler
   - Reshape into 24-hour windows
5. Run: `preds = model.predict(X_new_sc)`
6. Apply threshold: `active = (preds >= 0.35).astype(int)`

The `inference_config.json` contains all feature column names, scaling parameters, and the threshold value so you can reproduce this exactly.

---

**Q: Can I add a sixth disease?**

Yes, but it requires retraining the full model.  You would need to add the new disease to the `DISEASES` list in Section 3 and re-run all sections from the beginning.  The architecture automatically scales to handle `len(DISEASES)` streams and output heads.

---

**Q: What if some sensor columns are missing in my data?**

The column standardisation in Section 3 handles aliased names automatically.  If a sensor is completely absent, the feature engineering steps use `if col in df.columns` guards — missing sensors simply produce no features for that sensor, and the remaining features are still valid.  Model performance may degrade without key sensors like temperature and humidity.

---

**Q: How are the 30 output heads structured, exactly?**

The model has 5 diseases × 6 outputs each = 30 Keras output layers:

For each `disease` in `[early_blight, late_blight, leaf_mold, powdery_mildew, spider_mites]`:
- `inf_24h_{disease}` — infection % at 24h (0–1 scaled, multiply by 100 for actual %)
- `inf_48h_{disease}` — infection % at 48h
- `dlt_24h_{disease}` — net change at 24h (normalised; reverse with dlt_mean and dlt_std)
- `dlt_48h_{disease}` — net change at 48h
- `act_24h_{disease}` — probability that disease is active at 24h (apply `CLS_THR` for label)
- `act_48h_{disease}` — probability that disease is active at 48h

---

*Documentation generated for the AgriTwin-GH project — Tomato Disease Progression Model.*
*Notebook: `notebooks/tomato_disease_progression_model.ipynb`*
