# Tomato Growth Progression Model

> **Who is this for?**  
> This document is written for anyone — farmer, student, developer, or curious reader — with zero prior knowledge of machine learning or data science. Every concept is explained from the ground up, with analogies and plain language throughout.

---

## Table of Contents

1. [Why Does This Matter?](#1-why-does-this-matter)
2. [What Does This Model Actually Predict?](#2-what-does-this-model-actually-predict)
3. [The Six Growth Stages (Quick Recap)](#3-the-six-growth-stages-quick-recap)
4. [The Dataset — What Data Goes In?](#4-the-dataset--what-data-goes-in)
5. [How Does the Notebook Work? — A Plain-English Walkthrough](#5-how-does-the-notebook-work--a-plain-english-walkthrough)
   - [Section 0 — Run ID & Path Setup](#section-0--run-id--path-setup)
   - [Section 1 — Setup & Imports](#section-1--setup--imports)
   - [Section 2 — Data Loading](#section-2--data-loading)
   - [Section 3 — Data Standardisation](#section-3--data-standardisation)
   - [Section 4 — Cycle Integrity Checks](#section-4--cycle-integrity-checks)
   - [Section 5 — Progression Targets](#section-5--progression-targets)
   - [Section 6 — Feature Engineering](#section-6--feature-engineering)
   - [Section 7 — Sequence Building](#section-7--sequence-building)
   - [Section 8 — Train / Val / Test Split](#section-8--train--val--test-split)
   - [Section 9 — Feature Scaling](#section-9--feature-scaling)
   - [Section 10 — Baseline Models](#section-10--baseline-models)
   - [Section 11 — The LSTM Model](#section-11--the-lstm-model)
   - [Section 12 — Training the Model](#section-12--training-the-model)
   - [Section 13 — Evaluation](#section-13--evaluation)
   - [Section 14 — Inference Examples](#section-14--inference-examples)
   - [Section 15 — Visualisations](#section-15--visualisations)
   - [Section 16 — Saving All Artifacts](#section-16--saving-all-artifacts)
   - [Section 17 — Inference Utilities](#section-17--inference-utilities)
   - [Section 18 — Final Summary](#section-18--final-summary)
6. [What is an LSTM? (No maths required)](#6-what-is-an-lstm-no-maths-required)
7. [Multi-Task Learning — One Model, Five Answers](#7-multi-task-learning--one-model-five-answers)
8. [Baseline Models — How Do We Know the LSTM is Good?](#8-baseline-models--how-do-we-know-the-lstm-is-good)
9. [How We Measure Success](#9-how-we-measure-success)
10. [All Output Files — What Gets Saved and Where](#10-all-output-files--what-gets-saved-and-where)
11. [Running the Notebook](#11-running-the-notebook)
12. [End-to-End Flow Diagram](#12-end-to-end-flow-diagram)
13. [Common Questions (FAQ)](#13-common-questions-faq)
14. [Standalone Test Suite: `test_growth_stage_progression.py`](#14-standalone-test-suite-testgrowthstageprogressionpy)
15. [Glossary](#15-glossary)

---

## 1. Why Does This Matter?

A tomato plant does not grow randomly — it moves through a fixed sequence of biological stages, from a tiny seedling all the way to a ripe, harvestable fruit. At every stage along that journey, the plant has different needs:

- A **seedling** needs very gentle care and high humidity.
- A **flowering** plant needs precise temperature control and pollination assistance.
- An **unripe-fruit** plant needs calcium and potassium to build healthy fruit.
- A **ripe** plant requires immediate harvest before quality degrades.

**The problem:** In a large greenhouse, a grower cannot watch every plant every hour. By the time a human notices that a plant is transitioning into a critical stage, it may already be too late to respond optimally.

**The solution this model provides:** Using only the greenhouse's existing **sensor readings** (temperature, humidity, light, CO₂, etc.), the model:

1. Identifies what growth stage the plant is currently in.
2. Predicts what stage comes next.
3. Estimates **how many hours** until the transition happens.
4. Tells you whether that transition will happen **within 24 hours**.
5. Tells you whether that transition will happen **within 48 hours**.

This turns a greenhouse from a passive environment into a **proactive, predictive system** — giving operators time to prepare the right intervention *before* a critical stage transition occurs.

---

## 2. What Does This Model Actually Predict?

The model makes **five simultaneous predictions** for every hour of sensor data it receives:

| Output | Type | Plain-English Meaning |
|--------|------|-----------------------|
| **Current Stage** | Classification | "Right now this plant is in the *flowering* stage." |
| **Next Stage** | Classification | "After flowering, it will enter the *unripe* stage." |
| **Hours to Transition** | Regression (a number) | "The transition will happen in approximately 38.5 hours." |
| **Transition within 24h** | Binary (yes/no) | "Yes, there will be a stage change in the next 24 hours." |
| **Transition within 48h** | Binary (yes/no) | "Yes, there will be a stage change in the next 48 hours." |

All five predictions come from a **single model** in one pass — this is called *multi-task learning* (explained in [Section 7](#7-multi-task-learning--one-model-five-answers)).

---

## 3. The Six Growth Stages (Quick Recap)

The model works with six sequential stages. A tomato plant always passes through them in this exact order — it cannot skip or go backwards.

| # | Stage Name | Plain Description |
|---|------------|-------------------|
| 0 | `seedling` | Germinated seed, first tiny leaves |
| 1 | `early_vegetative` | True leaves growing, plant building structure |
| 2 | `flowering_initiation` | First flower buds becoming visible |
| 3 | `flowering` | Open yellow flowers, pollination occurring |
| 4 | `unripe` | Green fruits developing |
| 5 | `ripe` | Fruits colouring, ready to harvest |

> For a full description of each stage including visual features, management decisions, and environmental requirements, see the [Tomato Growth Stage Classification](TOMATO_GROWTH_STAGE_CLASSIFICATION.md) document.

---

## 4. The Dataset — What Data Goes In?

**File:** `data/processed/Growth Progression/tomato_growth_progression_synthetic_hourly.csv`

This is a **synthetic** (computer-generated, but physically realistic) hourly time-series dataset. "Hourly" means there is one data row for every hour of a tomato plant's life, across multiple complete growth cycles.

### What does each row contain?

| Column | What it means | Example value |
|--------|--------------|---------------|
| `timestamp` | The exact date and hour this reading was taken | `2025-03-01 14:00:00` |
| `cycle_id` | Which growth cycle this belongs to (each cycle = one full plant life) | `3` |
| `stage_name` | The growth stage label at this hour | `flowering` |
| `indoor_temp` (→ `temperature`) | Air temperature inside the greenhouse in °C | `22.4` |
| `indoor_humidity` (→ `humidity`) | Relative humidity in the greenhouse in % | `71.2` |
| `solarradiation` (→ `light`) | Light intensity reaching the plants | `345.0` |
| `indoor_CO2` (→ `co2`) | Carbon dioxide concentration in ppm | `812.0` |

> **Note:** Column names in the raw file vary (e.g., `indoor_temp` vs `temperature`). The notebook's standardisation step (Section 3) handles all these aliases automatically — you do not need to rename anything manually.

### What is a "growth cycle"?

Think of a **cycle** as one complete life of a single tomato plant — from germination all the way to harvest. The dataset contains **multiple cycles**, each representing a separate plant (or planting season). The model learns patterns from all cycles combined, then is asked to predict on cycles it has never seen before.

---

## 5. How Does the Notebook Work? — A Plain-English Walkthrough

The notebook is divided into **19 numbered sections** (Sections 0–18). Each section has a specific job. Here is what each one does and why.

---

### Section 0 — Run ID & Path Setup

**What it does:**  
This is the very first thing that runs. It implements an intelligent **Run ID detection** system:

1. **Scans existing artifacts** — Looks in `src/agritwin_gh/models/artifacts/` for directories matching the pattern `growth_stage_progression_*`
2. **Detects and reuses** — If existing model runs are found, extracts their Run IDs and reuses the latest one
3. **Fallback to new** — If no existing runs are found, generates a new Run ID (timestamp string like `20260310_142305`)

This means:
- Running the notebook multiple times uses the **same** trained model (no retraining)
- All predictions and outputs append to the existing artifact folder
- You never accidentally lose a previously trained model

**Why it matters:**  
Without Run ID detection, you would have to manually track which model was trained and either reuse it by hard-coding the path or retrain from scratch. The automatic detection makes the workflow reproducible and efficient.

**What it sets up:**

```
E:\AgriTwin-GH\
├── src\agritwin_gh\models\
│   ├── growth_stage_progression_<RUN_ID>.keras     ← the trained model
│   └── artifacts\growth_stage_progression_<RUN_ID>\
│       ├── plots\         ← all generated PNG charts
│       ├── metrics\       ← JSON/CSV metric files
│       ├── logs\          ← execution log file
│       └── reports\       ← text reports and prediction CSVs
```

All folder creation happens automatically — you do not need to create anything by hand.

**Implementation details:**
The detection scans the artifacts folder and looks for directories like `growth_stage_progression_20260310_192038`. When found, it extracts the Run ID (`20260310_192038`) and verifies the model file exists before reusing.

---

### Section 1 — Setup & Imports

**What it does:**  
Loads all the Python libraries (tools) the notebook needs and sets random seeds to make results **reproducible** — meaning if you run the notebook twice with the same data, you get the same results.

**Key libraries loaded:**

| Library | What it does (plain English) |
|---------|------------------------------|
| `pandas` | Works with tabular data (like Excel, but in Python) |
| `numpy` | Does fast numerical calculations on arrays of numbers |
| `matplotlib` / `seaborn` | Creates charts and plots |
| `scikit-learn` | Provides classic ML algorithms and preprocessing tools |
| `xgboost` | A powerful tree-based algorithm used for baseline comparison |
| `tensorflow` / `keras` | The deep learning framework used to build and train the LSTM |

Also defines the `save_fig()` helper — every plot in the notebook is automatically saved as a PNG file to the artifacts folder when this function is called.

---

### Section 2 — Data Loading

**What it does:**  
Reads the CSV dataset from disk and performs a quick health check — printing the shape (number of rows and columns), listing all column names, and counting missing values.

**Key output:** `dataset_summary.json` — a JSON file recording the dataset's basic statistics, saved to the artifacts folder as a permanent record of what data went into this run.

**What "loading" looks like (simplified):**

```python
# Read the file
df = pd.read_csv("...tomato_growth_progression_synthetic_hourly.csv")

# First 5 rows
df.head()
```

---

### Section 3 — Data Standardisation

**What it does:**  
Real datasets rarely arrive in perfect, consistent format. This section handles three problems:

1. **Column name aliases** — The raw file may call temperature `indoor_temp`, `air_temperature`, `temp_c`, or simply `temperature`. The notebook maps all known variants to a single canonical name automatically.

2. **Stage label normalisation** — The dataset might spell `early_vegetative` as `"early vegetative"` or `"Early Vegetative"`. All variants are mapped to the exact lowercase underscore form the model expects.

3. **Sorting and deduplication** — Rows are sorted chronologically within each cycle and any exact duplicates are removed.

**Why this matters:**  
If stage labels are inconsistent, the model might think `"Flowering"` and `"flowering"` are two different classes — which would corrupt the training data completely.

---

### Section 4 — Cycle Integrity Checks

**What it does:**  
Before training, the notebook validates every growth cycle to catch data quality problems. For each cycle it checks:

| Check | What it's looking for |
|-------|-----------------------|
| **Minimum rows** | The cycle must have at least 10 hourly readings |
| **Monotonic timestamps** | Time must always go forward, never backwards |
| **Duplicate timestamps** | No two rows with the same date-hour in one cycle |
| **Stage reversals** | A plant in `flowering` cannot go back to `seedling` |
| **Missing stages** | Each complete cycle should pass through all 6 stages |

**What happens to bad cycles?**  
- Cycles with **stage reversals** or **too few rows** are removed completely — they represent corrupted data that would teach the model wrong patterns.
- Cycles with **missing stages** (e.g., a cycle that was recorded starting from `flowering_initiation` rather than `seedling`) are flagged but kept — the model can still learn from partial cycles.

**Key output:** `cycle_summary.csv` — a CSV file with one row per cycle and a column listing any issues found.

---

### Section 5 — Progression Targets

**What it does:**  
This is where the five "answers" the model needs to learn are computed. For every row in the dataset, the code looks ahead in time (within the same cycle) to calculate:

- **`next_stage`** — What is the next stage the plant will enter?
- **`hours_to_next_stage`** — How many hours until that transition?
- **`transition_within_24h`** — Is that transition within 24 hours? (1 = yes, 0 = no)
- **`transition_within_48h`** — Is that transition within 48 hours? (1 = yes, 0 = no)
- **`stage_progress_fraction`** — How far through the current stage is the plant? (0.0 = just entered, 1.0 = about to leave)

**Analogy:** Imagine the plant's life as a road trip with 6 cities as stops. At any given point on that road, you can calculate: What city are you currently in? What's the next city? How many hours until you arrive? Will you get there in 24 hours? In 48 hours? How far through the current leg of the journey are you?

The last row in each stage (just before transitioning) will always have `NaN` (not a number) for `next_stage` and related columns, because there is no further transition to look forward to after entering the final `ripe` stage.

---

### Section 6 — Feature Engineering

**What it does:**  
The raw sensor readings (temperature, humidity, light, CO₂) are useful, but the model learns much better if it also has access to *derived* features — things like:

**Rolling statistics** — "What was the average temperature over the last 6 hours? 12 hours? 24 hours?" These capture trends that a single snapshot cannot.

**Lag features** — "What was the temperature 1 hour ago? 2 hours ago? 6 hours ago?" This gives the model a sense of direction — is the temperature rising or falling?

**Computed biologically-meaningful features:**

| Feature | Biological Meaning |
|---------|-------------------|
| `elapsed_hours` | How long has this cycle been running? |
| `hour_of_day` | What hour of the day is it (for circadian rhythm patterns)? |
| `cumulative_gdh` | **Growing Degree Hours** — cumulative heat units the plant has received (plants need a certain amount of heat to advance stages) |
| `cumulative_light` | How much total light has the plant received (important for flowering trigger) |
| `vpd` | **Vapour Pressure Deficit** — a measure of how effectively the plant can transpire; computed from temperature and humidity |

**Result:** Typically **60–80 features** per timestep, depending on how many raw sensor columns exist.

---

### Section 7 — Sequence Building

**What it does:**  
LSTM models do not take one row at a time — they take a **window of consecutive rows**.

Imagine sliding a 24-hour window across the timeline:
- Window 1: Hours 0–23 → predict what happens at hour 23
- Window 2: Hours 1–24 → predict what happens at hour 24
- Window 3: Hours 2–25 → predict what happens at hour 25
- ...and so on

Each window becomes one **training sample**. The model learns: "Given the last 24 hours of sensor readings, what growth state is the plant in and where is it going next?"

**The window size is `SEQ_LEN = 24`** (24 hours). This is a design choice — a 24-hour window captures a full day-night cycle, which is biologically meaningful for tomato growth.

**Output shape:**  
`X.shape = (N_samples, 24, N_features)` — a 3D array where:
- `N_samples` = total number of windows built
- `24` = time steps in each window
- `N_features` = number of features at each timestep

---

### Section 8 — Train / Val / Test Split

**What it does:**  
Divides the data into three sets. This is done at the **cycle level** (not the row level) to prevent data leakage.

| Set | Fraction | Purpose |
|-----|----------|---------|
| **Training** | 70% of cycles | The model learns from this data |
| **Validation** | 15% of cycles | Used during training to check for overfitting |
| **Test** | 15% of cycles | Held back completely until final evaluation |

**Why split by cycle, not by row?**  
If you split by row, sequences from the same cycle could appear in both training and test sets — the model would effectively be "previewing" the answer. Splitting by cycle ensures the model is evaluated on completely unseen plants.

**Key output:** `split_summary.json` — records exactly how many cycles and sequences are in each split.

---

### Section 9 — Feature Scaling

**What it does:**  
Raw sensor values have very different numerical ranges:
- Temperature: 15–35 (°C)
- CO₂: 400–1500 (ppm)
- Light: 0–1000 (W/m²)

Neural networks train much better when all inputs are on the same scale. A `StandardScaler` transforms every feature so that across the training set, each feature has a **mean of 0** and a **standard deviation of 1**.

**Analogy:** Imagine judging athletes where one runs 100m in 10 seconds and another lifts 200kg. The numbers mean completely different things. Scaling puts everything on a comparable "effort" scale.

**Important rule:** The scaler is **fitted only on training data** and then applied to validation and test data. This prevents information from the test set from leaking into the model during training.

**Key output:** `scaler_details.json` — records the mean and standard deviation for every feature so the scaler can be reconstructed later for inference.

---

### Section 10 — Baseline Models

**What it does:**  
Before training an expensive deep learning model, we establish what **simpler models** can achieve. If a simple model is nearly as good, we do not need the complex one.

Two baseline models are trained:

**Random Forest Classifier** — for predicting `current_stage`:
- Works by building many decision trees and combining their votes.
- Trained on the 2D (flattened) version of the same training sequences.

**XGBoost Regressor** — for predicting `hours_to_next_stage`:
- A gradient-boosting algorithm known for strong performance on tabular data.
- Also trained on the flattened sequences.

**Key outputs:**
- `baseline_metrics.json` — accuracy, MAE, RMSE of the baseline models
- `rf_feature_importances.csv` — which features the Random Forest found most predictive
- `baseline_rf_classification_report.txt` — precision, recall, F1 per stage class

The LSTM's performance is later compared to these baselines to quantify how much the temporal deep learning model adds.

---

### Section 11 — The LSTM Model

**What it does:**  
Builds the main deep learning model. The architecture has three parts:

**1. Shared LSTM Backbone:**

```
Input (24 timesteps × N features)
   ↓
LSTM Layer 1 (128 units, returns sequences)
   ↓
LSTM Layer 2 (64 units, outputs one vector)
   ↓
Batch Normalisation
   ↓
Dense Layer (64 units, ReLU activation)
   ↓
Dropout (30% of connections randomly disabled during training)
```

Think of the LSTM backbone as a "reader" that digests the last 24 hours of sensor history and compresses it into a single rich summary vector.

**2. Five Specialised Output Heads:**

After the shared backbone, five separate networks branch off, each specialised for one prediction task:

| Head | Output | Loss Function | Notes |
|------|--------|---------------|-------|
| `current_stage` | 6 probabilities (one per stage) | Categorical cross-entropy | What stage is the plant in *now*? |
| `next_stage` | 6 probabilities (one per stage) | Categorical cross-entropy | What stage comes *after* current? |
| `hours_to_next` | A single number (hours) | Huber loss (robust to outliers) | How many hours until transition? |
| `trans_24h` | A probability between 0 and 1 | Binary cross-entropy | Will transition occur in next 24h? |
| `trans_48h` | A probability between 0 and 1 | Binary cross-entropy | Will transition occur in next 48h? |

**3. Class Weighting:**  
Some growth stages (like `ripe`) have fewer hours than others (like `early_vegetative`). Without correction, the model would learn to mostly predict the majority stages. Class weights are computed and applied so that the model pays proportionally more attention to rare stages during training.

**4. Loss Function Configuration:**

The model is trained with a multi-task loss function that includes all five outputs:

```python
loss = {
    'current_stage': 'categorical_crossentropy',
    'next_stage': 'categorical_crossentropy',
    'hours_to_next': 'mse',
    'trans_24h': 'mse',
    'trans_48h': 'mse',
}
```

Each loss component is weighted equally during backpropagation. This ensures all five tasks learn simultaneously and reinforce each other.

**Key outputs:**
- `model_config.json` — all hyperparameters (architecture choices, layer sizes, learning rate)
- `class_weights.json` — the weight given to each stage class

---

### Section 12 — Training the Model

**What it does:**  
Runs the actual training loop — the model sees the training data repeatedly (one pass through all data = one "epoch") and adjusts its internal parameters to improve its predictions.

**Training safeguards:**

| Callback | What it does |
|----------|-------------|
| **EarlyStopping** | Stops training automatically if the model is no longer improving on the validation set (prevents wasted compute and overfitting) |
| **ReduceLROnPlateau** | Reduces the learning rate if progress stalls — like taking smaller steps when you're close to the answer |
| **ModelCheckpoint** | Saves the single best model weights seen during training (even if later epochs get worse) |

**Data pipeline:** A `tf.data.Dataset` pipeline is used to feed data to the model in efficiently batched, shuffled form — the training data is shuffled every epoch so the model cannot memorise order.

**Key outputs:**
- `training_history.csv` and `training_history.json` — full record of loss and accuracy for every epoch
- `training_history_curves.png` — a plot showing how all metrics evolved during training

---

### Section 13 — Evaluation

**What it does:**  
Runs the trained model on the **test set** (data it has never seen) and computes all performance metrics.

**For `current_stage` and `next_stage` classification:**
- **Accuracy** — percentage of correct predictions
- **Precision** — of all the times the model predicted "flowering", how often was it actually flowering?
- **Recall** — of all the actual "flowering" sequences, how many did the model find?
- **F1 Score** — harmonic mean of precision and recall (balances both)
- **Confusion Matrix** — a grid showing which stages got confused with which others

**For `hours_to_next_stage` regression:**
- **MAE** (Mean Absolute Error) — average number of hours off per prediction
- **RMSE** (Root Mean Squared Error) — penalises large errors more heavily
- **MAPE** — Mean Absolute Percentage Error — "on average, predictions are X% off"

**For `transition_within_24h` and `transition_within_48h`:**
- **Accuracy**, **Precision**, **Recall**, **F1 Score**
- **ROC-AUC** — how well the model separates "will transition" from "won't transition"

**Key outputs:**
- `confusion_matrix_current_stage.png` and `confusion_matrix_next_stage.png`
- `confusion_matrix_trans_24h.png` and `confusion_matrix_trans_48h.png`
- `lstm_current_stage_classification_report.txt`
- `lstm_next_stage_classification_report.txt`
- `hours_actual_vs_predicted.png` — scatter plot of predicted vs actual hours
- `lstm_metrics.json` — all computed metrics in one file

---

### Section 14 — Inference Examples

**What it does:**  
Picks 8 random samples from the test set and runs the model on each, printing side-by-side comparisons of predicted vs actual values. This gives a human-readable sanity check — you can read individual examples to build intuition for how the model behaves.

**Critical logic: Conditional Transition Flags**

The inference function implements a crucial safeguard for the transition probability outputs:

```python
# If hours to next stage > 24, set trans_24h to 0; otherwise use predicted probability
trans_24h_prob = 0.0 if hours_pred > 24 else raw_trans_24h

# If hours to next stage > 48, set trans_48h to 0; otherwise use predicted probability  
trans_48h_prob = 0.0 if hours_pred > 48 else raw_trans_48h
```

This ensures the transition flags are *conditional probabilities* — they only produce meaningful values when the time window is biologically relevant. For example:
- If `hours_to_next = 220h`, both `trans_24h` and `trans_48h` will be set to 0.0 (no chance of transition in those windows)
- If `hours_to_next = 18h`, `trans_24h` will use the model's prediction (likely high), and `trans_48h` will also use its prediction (likely very high)
- If `hours_to_next = 50h`, `trans_24h` will be 0.0, and `trans_48h` will use its prediction

**Example output row (correct logic in action):**

```
Sample #3
  Actual current stage      : flowering
  Predicted current stage   : flowering  ✓
  Actual next stage         : unripe
  Predicted next stage      : unripe     ✓
  Actual hours to next      : 41.0 h
  Predicted hours to next   : 38.7 h     (Δ 2.3 h)
  Trans prob 24h (raw)      : 0.03   → Output: 0.0 (hours > 24)
  Trans prob 48h (raw)      : 0.87   → Output: 0.87 (hours < 48) ✓
```

Notice how the raw model outputs are post-processed based on `hours_to_next` to produce biologically sensible predictions.

**Key output:** `prediction_samples.csv` — a CSV file with all 8 example predictions for later review.

---

### Section 15 — Visualisations

**What it does:**  
Generates 5 publication-quality charts for understanding the dataset and model behaviour:

| Plot | What it shows |
|------|--------------|
| `stage_distribution.png` | How many hours of data exist per stage (are classes balanced?) |
| `cycle_duration_distribution.png` | How long each complete growth cycle lasted (in hours) |
| `stage_progression_timelines.png` | A timeline view of 6 random cycles showing stage transitions |
| `transition_probability_histograms.png` | Distribution of how often 24h and 48h transitions occur |
| `rf_feature_importances.png` | Which sensor features matter most (from the Random Forest) |

All plots are saved as PNG files in the artifacts folder.

---

### Section 16 — Saving All Artifacts

**What it does:**  
Persists all key outputs of this run to disk in an organised way.

Specifically:
1. **Saves the trained Keras model** → `growth_stage_progression_<RUN_ID>.keras` — can be loaded later for inference without retraining.
2. **Saves the feature scaler** → `feature_scaler.pkl` — required to preprocess new sensor data in exactly the same way as training data.
3. **Saves the inference config** → `inference_config.json` — a single file containing all information needed to run the model on new data: feature list, stage mappings, scaling parameters, sequence length.
4. **Saves per-stage class distributions** → `per_stage_distribution.json`
5. **Saves full test set predictions** → `test_predictions_full.csv` — every test sequence with all 5 predicted and actual values.

---

### Section 17 — Inference Utilities

**What it does:**  
Defines five reusable Python functions that make it easy to use the trained model on **new, unseen sensor data** — without having to understand the internal preprocessing pipeline:

| Function | What it does |
|----------|----------|
| `decode_stage(idx)` | Converts a stage number (0–5) back to its name (`"flowering"`) |
| `preprocess_new_data(df_raw)` | Takes a raw sensor DataFrame and runs the full pipeline (standardise columns → engineer features → fill gaps) |
| `build_recent_sequence(df_feats)` | Takes the last 24 rows of preprocessed data and returns a scaled array ready for the model |
| `predict_with_inference(model, sample_data_dict, scaler)` | Runs the model on a sample and applies **conditional transition flag logic** (see below) |
| `predict_progression(sequence_3d)` | Runs the model and returns a human-readable dict with all 5 predictions |

**Conditional Transition Flag Processing:**

The `predict_with_inference` function applies critical post-processing to ensure stable, biologically valid predictions:

```python
def predict_with_inference(model, sample_data_dict, scaler):
    # Get raw model outputs
    sample_scaled = scaler.transform(sample_data_dict['X'])
    preds = model.predict(sample_scaled, verbose=0)
    
    hours_pred = preds[2][0]
    raw_trans_24h = preds[3][0]
    raw_trans_48h = preds[4][0]
    
    # Apply conditional logic: transition flags should be ~0 if hours > threshold
    trans_24h_prob = 0.0 if hours_pred > 24 else raw_trans_24h
    trans_48h_prob = 0.0 if hours_pred > 48 else raw_trans_48h
    
    return trans_24h_prob, trans_48h_prob  # Now logically valid
```

This ensures that:
- If a plant has 220 hours until the next stage transition, the 24h and 48h flags will output 0.0 (impossible to transition in those windows)
- Transition flags only produce meaningful probabilities when `hours_to_next` is actually within their respective windows
- Predictions are consistent with each other — no contradictions like "transition in 220 hours but 50% chance in 24 hours"

A **demonstration** is run at the end using the first test sequence, printing the results and saving them to `inference_demo.json`.

---

### Section 18 — Final Summary

**What it does:**  
Computes and prints a complete summary of the run — the final performance numbers, how much the LSTM improved over the baselines, and a listing of every artifact produced. Saved as both a machine-readable JSON file and a human-readable text file.

**Key outputs:**
- `run_summary.json` — all metrics and metadata in structured format
- `run_summary.txt` — a formatted text table of the same information

---

## 6. What is an LSTM? (No maths required)

**LSTM** stands for *Long Short-Term Memory*. It is a type of neural network specifically designed to work with **sequences** — data where the order matters.

**Analogy:** Think of a person reading a story. When they encounter the word "but", they know it reverses the meaning of what was just said. They remember the previous sentences to understand each new word. An LSTM works the same way:

- It reads sensor readings one hour at a time, in order.
- It maintains an internal "memory" that captures what has happened so far.
- When it reaches the last hour of the 24-hour window, its memory contains a compressed understanding of the entire recent history.
- This rich memory is then used to make predictions.

Standard neural networks cannot do this — they treat each input independently, without any notion of "what came before". LSTMs were designed specifically to capture these temporal dependencies.

**What "long short-term memory" means:**
- **Short-term memory** — the hidden state: what the network "just thought about"
- **Long-term memory** — the cell state: context it has decided to preserve for a long time
- The LSTM learns *which information to remember and which to forget* using special internal "gates"

---

## 7. Multi-Task Learning — One Model, Five Answers

Traditional machine learning trains a separate model for each question. This notebook uses **multi-task learning** — one model that answers all five questions simultaneously.

**How it works:** The LSTM backbone is shared — it learns a general understanding of tomato growth dynamics from all five tasks at once. Then five small specialised "heads" branch off the end, each fine-tuned for one specific prediction.

**Why is this better than 5 separate models?**

1. **More data efficiency** — The backbone learns from all five signals at once. The `hours_to_next_stage` regression task provides gradient signal that also helps the stage classification tasks, and vice versa.

2. **Shared knowledge** — Understanding that a plant is likely to transition soon (from the 24h/48h heads) naturally reinforce the hours-to-next regression head, and vice versa. The tasks are related and learning them together improves all of them.

3. **Single inference** — At prediction time, you run the model once and get all five answers — much faster and simpler than running five separate models.

---

## 8. Baseline Models — How Do We Know the LSTM is Good?

A baseline model is the simplest reasonable approach we can compare against. If the LSTM does not significantly outperform the baseline, it was not worth the added complexity.

This notebook uses two baselines:

**Random Forest for stage classification:**
- A Random Forest trains dozens of decision trees, each seeing a random subset of the data.
- Each tree votes on the stage, and the majority vote wins.
- It does not understand sequences or temporal order — it treats all 24 × N_features values as flat, independent features.

**XGBoost for hours-to-transition regression:**
- XGBoost (eXtreme Gradient Boosting) builds trees sequentially, each one correcting the errors of the previous.
- Again, it sees the flattened sequence without any temporal understanding.

**Why these baselines matter:** If the LSTM's MAE for hours-to-transition is 5.2h compared to the XGBoost baseline's 8.7h, we know the temporal learning is providing a real improvement of 3.5 hours per prediction — a meaningful operational advantage.

---

## 9. How We Measure Success

### For Stage Classification (current stage, next stage)

**Accuracy** — The simplest metric. If the model correctly classifies 92 out of 100 sequences, accuracy = 92%.

**Confusion Matrix** — A table showing what the model predicted vs what was actually true. Each row is an actual class, each column is a predicted class. Diagonal entries are correct predictions; off-diagonal entries are mistakes. This shows *which* stages get confused with which — far more informative than a single number.

**F1 Score** — Since some stages have fewer examples than others (class imbalance), accuracy alone is misleading. F1 balances precision and recall and is reported per-class and as a weighted average.

### For Hours Regression

**MAE (Mean Absolute Error)** — "On average, predictions are X hours off." MAE = 3.5 means predictions are 3.5 hours off on average. Easy to interpret.

**RMSE (Root Mean Squared Error)** — Similar to MAE but squares the errors before averaging, meaning large errors are penalised disproportionately. RMSE > MAE indicates there are occasional very large prediction errors.

**MAPE (Mean Absolute Percentage Error)** — Errors expressed as a percentage. MAPE = 12% means predictions are 12% off relative to the actual value.

### For Transition Probability (24h / 48h)

**F1 Score** — Whether the plant is within 24 hours of transitioning is a yes/no question. F1 score measures how well the model identifies the positive case (transition will happen).

**ROC-AUC** — A threshold-independent measure of how well the model's probability score separates "will transition" from "won't transition". Perfect = 1.0, random = 0.5.

---

## 10. All Output Files — What Gets Saved and Where

After a complete run, all outputs live under:

```
E:\AgriTwin-GH\src\agritwin_gh\models\artifacts\growth_stage_progression_<RUN_ID>\
```

### Metrics (`metrics/`)

| File | What it contains |
|------|-----------------|
| `dataset_summary.json` | Row count, column list, missing values |
| `cycle_summary.csv` | Per-cycle quality audit results |
| `feature_list.json` | All feature names, rolling windows, lag steps |
| `split_summary.json` | Train/val/test cycle and sequence counts |
| `scaler_details.json` | Mean and std for every feature |
| `baseline_metrics.json` | RF accuracy, XGBoost MAE and RMSE |
| `rf_feature_importances.csv` | Feature importance from the Random Forest |
| `class_weights.json` | Per-stage class weights and training distribution |
| `model_config.json` | All model hyperparameters |
| `training_history.csv` | Loss and accuracy per epoch |
| `training_history.json` | Same, in JSON format |
| `lstm_metrics.json` | All final test-set evaluation metrics |
| `per_stage_distribution.json` | Stage class counts across train/val/test |

### Reports (`reports/`)

| File | What it contains |
|------|-----------------|
| `baseline_rf_classification_report.txt` | Per-class precision/recall/F1 for Random Forest |
| `lstm_current_stage_classification_report.txt` | Per-class precision/recall/F1 for LSTM current-stage head |
| `lstm_next_stage_classification_report.txt` | Per-class precision/recall/F1 for LSTM next-stage head |
| `prediction_samples.csv` | 8 hand-picked inference examples with predictions |
| `test_predictions_full.csv` | All test sequences with all 5 predicted + actual values |
| `inference_demo.json` | Output of the inference demo call |

### Plots (`plots/`)

| File | What it shows |
|------|--------------|
| `training_history_curves.png` | Loss and accuracy curves during training |
| `confusion_matrix_current_stage.png` | Stage classification confusion matrix |
| `confusion_matrix_next_stage.png` | Next-stage prediction confusion matrix |
| `confusion_matrix_trans_24h.png` | 24h transition binary classification CM |
| `confusion_matrix_trans_48h.png` | 48h transition binary classification CM |
| `hours_actual_vs_predicted.png` | Scatter: actual vs predicted hours to transition |
| `stage_distribution.png` | Dataset class balance visualisation |
| `cycle_duration_distribution.png` | Distribution of cycle lengths |
| `stage_progression_timelines.png` | Stage-over-time plots for sample cycles |
| `transition_probability_histograms.png` | 24h/48h transition distribution |
| `rf_feature_importances.png` | Random Forest feature importance bar chart |

### Run-Level (artifacts root)

| File | What it contains |
|------|-----------------|
| `best_model_<RUN_ID>.keras` | Best checkpoint saved during training |
| `feature_scaler.pkl` | Serialised StandardScaler for inference |
| `inference_config.json` | All config needed to run inference on new data |
| `run_summary.json` | Complete run metadata and final metrics |
| `run_summary.txt` | Human-readable summary table |
| `run_<RUN_ID>.log` | Full execution log with timestamps |

### Model Save (parent models folder)

| File | What it contains |
|------|-----------------|
| `growth_stage_progression_<RUN_ID>.keras` | Final saved trained model |

---

## 11. Running the Notebook

### Prerequisites

Ensure the project virtual environment is active and all requirements are installed:

```bash
# From E:\AgriTwin-GH\
.venv\Scripts\activate     # Windows
# or
source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### Steps

1. Open the notebook in VS Code or Jupyter:
   ```
   notebooks/tomato_growth_stage_progression.ipynb
   ```

2. Select the project virtual environment kernel (`.venv`).

3. Run all cells top to bottom using **Run All** (`Ctrl+F9` in VS Code).

4. The notebook will:
   - Generate a new `RUN_ID` automatically
   - Create all output folders
   - Run all 19 sections sequentially
   - Take approximately 15–45 minutes (depending on hardware and GPU availability)

5. After completion, find all results under:
   ```
   src/agritwin_gh/models/artifacts/growth_stage_progression_<RUN_ID>/
   ```

### Running on GPU

TensorFlow automatically uses a CUDA-compatible GPU if one is available and the CUDA drivers are installed. The notebook confirms GPU availability in Section 1:
```
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
If the list is empty, training runs on CPU — still correct, but slower.

---

## 12. End-to-End Flow Diagram

```
CSV Dataset
    │
    ▼
[Section 2] Load CSV
    │
    ▼
[Section 3] Standardise Columns & Stage Labels
    │
    ▼
[Section 4] Cycle Integrity Checks → Remove Bad Cycles
    │
    ▼
[Section 5] Compute 5 Target Variables per Row
    │  (next_stage, hours_to_next, trans_24h, trans_48h, stage_frac)
    ▼
[Section 6] Feature Engineering → 60-80 Features per Hour
    │  (rolling stats, lag features, GDH, VPD, light cumsum)
    ▼
[Section 7] Slide 24-Hour Windows → 3D Tensor (N, 24, F)
    │
    ▼
[Section 8] Split by Cycle → Train / Val / Test
    │
    ▼
[Section 9] StandardScaler (fit on train only)
    │
    ├──▶ [Section 10] Baselines (RF + XGBoost) → baseline_metrics.json
    │
    ▼
[Section 11] Build LSTM Multi-Task Model (5 heads)
    │
    ▼
[Section 12] Train with EarlyStopping + ReduceLR + Checkpoint
    │           → training_history_curves.png
    ▼
[Section 13] Evaluate on Test Set
    │           → confusion matrices, classification reports, lstm_metrics.json
    ▼
[Section 14] Print 8 Inference Examples → prediction_samples.csv
    │
    ▼
[Section 15] Generate 5 Dataset & Model Visualisation Plots
    │
    ▼
[Section 16] Save Model (.keras) + Scaler (.pkl) + Config (JSON)
    │
    ▼
[Section 17] Define Inference Utilities + Run Demo
    │
    ▼
[Section 18] Compute & Save Final Run Summary
                → run_summary.json + run_summary.txt
```

---

## 13. Common Questions (FAQ)

**Q: Can I use my own real greenhouse sensor data instead of the synthetic dataset?**  
A: Yes. As long as your CSV has a `timestamp` column, a growth cycle identifier, a stage label column, and at least one environmental sensor column (temperature, humidity, light, or CO₂), the notebook will handle it. Column names do not need to match exactly — the alias system in Section 3 maps common variants automatically.

**Q: What if my data does not have all six growth stages?**  
A: The notebook flags cycles with missing stages but keeps them for training. You can also adjust the `REQUIRE_START_STAGE` and `REQUIRE_END_STAGE` constants in Section 4 to change which stages are required.

**Q: How long does training take?**  
A: On a modern GPU, approximately 5–15 minutes with early stopping. On CPU, 30–90 minutes depending on dataset size.

**Q: Can I re-use a model trained in a previous run without retraining?**  
A: Yes. Use the `inference_config.json` file from the artifacts folder to load the scaler and model paths, then use the inference utilities from Section 17. All the information needed to reconstruct the inference pipeline is saved.

**Q: What does the `solarradiation` column get renamed to?**  
A: It gets renamed to `light` by the column alias map in Section 3. All such renamings are printed during the "Detected columns" step.

**Q: Why does the last row of each stage have `NaN` for some target columns?**  
A: Because `hours_to_next_stage`, `transition_within_24h`, and `transition_within_48h` require looking ahead in time. For rows that are at the very last recorded stage (`ripe`), there is no future transition to look forward to, so these values are `NaN`. The model handles this with sample weights — NaN rows are given zero weight during training and are excluded from evaluation metrics.

**Q: What is the difference between the `.keras` checkpoint and the final model?**  
A: The checkpoint (`best_model_<RUN_ID>.keras`) is saved by the `ModelCheckpoint` callback and contains the weights from the single best validation epoch — even if later epochs were worse. The final model (`growth_stage_progression_<RUN_ID>.keras`) is saved at the end of Section 16 and contains whatever state the model was in when training finished. In most cases these are identical, but if you use the saved model for deployment, prefer the checkpoint.

---

## 14. Standalone Test Suite: `test_growth_stage_progression.py`

### 14.1 Overview

**File location:** `scripts/test_growth_stage_progression.py`

**Purpose:**  
Standalone test script to validate the trained Growth Stage Progression model (multi-task LSTM) across 10 diverse scenarios covering all six growth stages, transition boundaries, stress conditions, and day/night comparisons.

**Why it exists:**  
The model makes five simultaneous predictions (current stage, next stage, hours-to-transition, 24h probability, 48h probability). This script exercises the model without requiring the training notebook or integration with the full digital twin — enabling quick validation, debugging, and confidence checks.

### 14.2 Usage

```bash
# Run all 10 scenarios
python scripts/test_growth_stage_progression.py

# Run a specific scenario (1–10)
python scripts/test_growth_stage_progression.py --scenario 5
```

### 14.3 What the Script Tests

| # | Scenario | What it validates |
|---|----------|-------------------|
| 1 | **Seedling Day 1** – freshly transplanted | Model correctly identifies stage 0 (seedling) at cycle onset |
| 2 | **Seedling near transition** – 90% progress, ~20h to next | High `t24_prob` expected; model should detect imminent transition |
| 3 | **Early Vegetative** – stable mid-stage (50% progress) | Low transition probabilities; model should predict stable state |
| 4 | **Flowering Initiation** – first flower buds appearing | Model correctly identifies stage 2 (flowering initiation) |
| 5 | **Full Flowering** – optimal conditions, peak anthesis | Model identifies stage 3 (flowering) with moderate hrs_to_next |
| 6 | **Unripe → Ripe transition** – 85% progress, 15h remaining | High `t24_prob` and `t48_prob`; imminent stage transition |
| 7 | **Ripe final phase** – 95% cycle progress | Model identifies stage 5 (ripe); `t24_prob` and `t48_prob` should be ≈0 |
| 8 | **Cold stress** – Early Veg at 10°C + low light | Model should predict slower development (higher hrs_to_next vs warm scenario) |
| 9 | **Heat stress** – Flowering at 38°C (pollen viability risk) | Model should detect stress condition; may affect transition timing |
| 10 | **Day vs Night** – same Flowering stage, toggle day/night flag | Comparison: daytime vs nighttime should show model responsiveness to diurnal cycle |

### 14.4 Expected Output Structure

For each scenario, the script prints:

```
======================================================================
Scenario  5: Full Flowering — mid-stage optimal conditions
  Current stage      : Stage 4 – Flowering
  Next stage         : Stage 5 – Unripe
  Hrs to transition  : 248.3 h  (~10.3 days)
  Transition in 24h  : 0.015  
  Transition in 48h  : 0.042  ##
```

**Interpretation:**
- `Current stage` — The model's classification of the current 24-timestep sequence
- `Next stage` — Predicted next stage (deterministic: always follows the sequence order)
- `Hrs to transition` — Regression output: minutes/hours until the transition occurs (may be negative if in final stage)
- `Transition in 24h` — Probability (0–1) that a transition occurs within 24 hours
- `Transition in 48h` — Probability (0–1) that a transition occurs within 48 hours

**Validation tips:**
- Scenarios 2, 6: expect high `t24_prob` (≥ 0.7) since they model near-transition conditions
- Scenario 3: expect low transition probabilities (~0.0–0.1) since it's mid-stage
- Scenario 7 (Ripe): both probabilities should be ~0 (no stage after ripe except harvest)
- Scenarios 8, 9: compare against corresponding normal scenarios to validate stress response
- Scenario 10: day/night outputs should be similar in structure but may differ in hrs_to_next due to environmental differences

### 14.5 Feature Input Strategy

Each scenario constructs a static 24-timestep sequence where:
- **All 24 timesteps** have **identical feature values** (snapshot model: no temporal variation within the sequence)
- Primary features (indices 0–32) are set based on scenario parameters (stage, progress, temperature, etc.)
- Rolling/lag features (indices 33–357) copy primary values into 11 slots per feature (consistent with training scaler)

**Key features set per scenario:**

- `stage_index` (feat 6) – numeric stage ID (0–5)
- `hours_in_current_stage` (feat 7) – how long plant has been in current stage
- `stage_progress_pct` (feat 11) – 0–100 representing where in the stage cycle the plant is
- `estimated_hrs_to_next_stage` (feat 14) – model's target for regression
- `indoor_temp`, `humidity`, `solarradiation`, `vpd` (feats 16, 17, 20, 22) – environmental drivers
- `day_night_flag` (feat 21) – 1.0 for day, 0.0 for night

### 14.6 Troubleshooting Failed Scenarios

**All predictions are "ripe" (Stage 6):**
- Check that the model file exists at the path printed in the load phase
- Verify the scaler is loaded correctly (should show `n_features_in_: 358`)
- Known issue: verify model checkpoint is not corrupted (retrain if needed)

**Import errors (TensorFlow, joblib, etc.):**
- Confirm virtual environment is activated
- Re-install dependencies: `pip install -r requirements.txt`

**"Model input shape mismatch" errors:**
- Verify all 24 timesteps are filled correctly (check `_fill_sequence()` output shape)
- Confirm feature count is exactly 358 after scaler transformation

**Unexpected transition probabilities (e.g., high `t24_prob` at seedling day 1):**
- This suggests model was trained on different feature distributions; compare scenario feature values against training dataset statistics
- Regenerate the model if dataset or feature engineering logic changed

### 14.7 Integration with AgriTwin-GH

This script is a **standalone diagnostic tool** — it does not interface with the REST API, database, or digital twin renderer. It is used for:

1. **Model validation** – After training, before deploying to the greenhouse control system
2. **Feature debugging** – Check whether feature fill logic produces sensible model outputs
3. **Rapid iteration** – Test model changes without restarting the full inference pipeline
4. **Documentation** – Provides clear examples of how to construct sequences for inference

For live greenhouse deployment, sensor data flows through `src/agritwin_gh/models/growth_stage_inference.py` → REST API → digital twin.

---

## 15. Glossary

| Term | Plain-English Definition |
|------|--------------------------|
| **Accuracy** | Percentage of correct predictions out of all predictions |
| **AUC / ROC-AUC** | A measure of how well a binary classifier separates positive and negative cases; 1.0 is perfect, 0.5 is random |
| **Batch** | A small group of samples processed together during training (more efficient than one at a time) |
| **Baseline model** | A simple comparison model — if your complex model cannot beat it, the complex model is probably not worth using |
| **Callback** | A function that runs automatically at certain points during training (e.g., after each epoch) |
| **Class imbalance** | When some categories have many more examples than others; this can cause a model to ignore rare classes |
| **Class weight** | A multiplier applied during training to compensate for class imbalance — rare classes get a higher weight |
| **CO₂ (ppm)** | Carbon dioxide concentration in parts per million; plants absorb CO₂ for photosynthesis |
| **Confusion matrix** | A grid showing actual vs predicted class labels; diagonal = correct, off-diagonal = errors |
| **Cross-entropy** | A loss function used for classification tasks; measures how wrong probability predictions are |
| **Cycle** | One complete growth period of a tomato plant, from germination to harvest |
| **Deep learning** | Machine learning using neural networks with many layers |
| **Dropout** | A training technique where random connections are disabled during each training step, preventing over-reliance on any single feature |
| **Early stopping** | A training safeguard that halts training when validation performance stops improving |
| **Epoch** | One complete pass through the entire training dataset |
| **F1 Score** | Harmonic mean of precision and recall; useful when classes are imbalanced |
| **Feature** | A single measurable property used as input to the model (e.g., temperature, humidity) |
| **Feature engineering** | Computing new derived features from raw data that help the model learn better |
| **Feature importance** | A measure of how much each input feature contributes to a model's predictions |
| **GDH (Growing Degree Hours)** | Cumulative heat above a base temperature (10°C); a biological "thermal clock" for plant development |
| **Gradient** | The signal used to update model weights during training; shows which direction to adjust parameters |
| **Huber loss** | A regression loss function that behaves like MAE for large errors and MSE for small ones — robust to outliers |
| **Lag feature** | The value of a variable from a previous timestep (e.g., temperature 3 hours ago) |
| **LSTM** | Long Short-Term Memory — a type of recurrent neural network designed to capture long-range patterns in sequential data |
| **MAE** | Mean Absolute Error — average of |predicted − actual| |
| **MAPE** | Mean Absolute Percentage Error — errors as a percentage of actual values |
| **Multi-task learning** | Training one model to predict multiple outputs simultaneously, sharing knowledge between tasks |
| **NaN** | "Not a Number" — a placeholder for missing or undefined values |
| **Overfitting** | When a model memorises the training data and performs poorly on new, unseen data |
| **Precision** | Of all times the model predicted "positive", what fraction were actually positive? |
| **Recall** | Of all actual positives, what fraction did the model correctly identify? |
| **Regression** | Predicting a continuous numerical value (e.g., hours until a transition) |
| **RMSE** | Root Mean Squared Error — like MAE but penalises large errors more heavily |
| **Rolling statistics** | Statistics (mean, std) computed over a sliding time window |
| **Run ID** | A unique timestamp string appended to all output file names so multiple runs do not overwrite each other |
| **Scaler** | A preprocessing tool that normalises feature values to a common scale |
| **Sequence** | An ordered series of sensor readings over time (here: 24 consecutive hourly readings) |
| **Sigmoid** | An S-shaped activation function that squashes output to the range [0, 1]; used for probability predictions |
| **Softmax** | An activation function that converts raw scores into probabilities that sum to 1; used for multi-class classification |
| **TFT** | Temporal Fusion Transformer — an alternative temporal sequence model (not used here, but referenced in other notebooks) |
| **VPD (Vapour Pressure Deficit)** | A measure of how "thirsty" the air is; high VPD drives more plant transpiration and can cause stress |
| **Validation set** | Data held out during training (not used for weight updates) to monitor for overfitting |
| **Window / Sliding window** | A fixed-length subsequence extracted by sliding a frame across a time series |
| **XGBoost** | eXtreme Gradient Boosting — a powerful, widely-used tree-based machine learning algorithm |
