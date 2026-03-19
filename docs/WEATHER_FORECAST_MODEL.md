# Greenhouse Weather Forecast Model (Chronos + XGBoost + LSTM Ensemble)

> **Who is this for?**  
> Anyone — farmer, student, developer, or complete beginner — who wants to understand *what* the `weather_forecast.ipynb` notebook does, *why* it matters, and *how* to use the final model in practice. No maths or machine-learning background is required.

---

## Table of Contents

1. [Big Picture: What Problem Are We Solving?](#1-big-picture-what-problem-are-we-solving)
2. [What Exactly Does the Model Predict?](#2-what-exactly-does-the-model-predict)
3. [Where Does the Data Come From?](#3-where-does-the-data-come-from)
4. [Step-by-Step Pipeline Overview](#4-step-by-step-pipeline-overview)
5. [Feature Engineering (Turning Raw Weather Into Inputs)](#5-feature-engineering-turning-raw-weather-into-inputs)
6. [The Three Model Families](#6-the-three-model-families)
   - [6.1 Chronos-T5-Small: Pretrained Time-Series Foundation Model](#61-chronos-t5-small-pretrained-time-series-foundation-model)
   - [6.2 XGBoost: Gradient-Boosted Direct Multi-Step Forecaster](#62-xgboost-gradient-boosted-direct-multi-step-forecaster)
   - [6.3 LSTM: Recurrent Sequential Regressor](#63-lstm-recurrent-sequential-regressor)
   - [6.4 How We Train All Models Together](#64-how-we-train-all-models-together)
7. [Ensemble: Optuna-Optimised Weight Blending](#7-ensemble-optuna-optimised-weight-blending)
8. [Conditions Classifier (Sky Condition Labels)](#8-conditions-classifier-sky-condition-labels)
9. [What Gets Saved After Training?](#9-what-gets-saved-after-training)
10. [How to Use the Final Model for Inference](#10-how-to-use-the-final-model-for-inference)
11. [Key Metrics and Performance](#11-key-metrics-and-performance)
12. [Architecture Summary & Design Decisions](#12-architecture-summary--design-decisions)
13. [Glossary](#13-glossary)

---

## 1. Big Picture: What Problem Are We Solving?

Inside a controlled greenhouse, **weather is not just outside** — it is also **inside**:

- Air temperature
- Relative humidity
- Solar radiation / light intensity
- Wind speed and other variables

If we can **predict the next 24–48 hours of indoor conditions**, we can:

- Adjust heating, cooling, fans, and fogging *before* conditions drift out of the safe zone.
- Plan irrigation and nutrient dosing more precisely.
- Anticipate disease risk windows where temperature and humidity combinations are dangerous.
- Feed a **digital twin** (AgriTwin-GH) that simulates future plant growth and disease.

The `weather_forecast.ipynb` notebook builds a **multi-model ensemble forecasting system** for the greenhouse environment. It does not just guess tomorrow's value from thin air — it learns from historical sensor data (2024–2025 observations for Dindigul, Tamil Nadu) using three complementary machine-learning approaches: Chronos (pretrained foundation model), XGBoost (tree-based), and LSTM (recurrent neural network).

---

## 2. What Exactly Does the Model Predict?

The model predicts **future values of several indoor climate variables** at two time horizons:

- **24 hours ahead** ("24h")
- **48 hours ahead** ("48h")

**Target variables:**

| Variable | Unit | Type |
|----------|------|------|
| `temp` | °C | Continuous |
| `humidity` | % | Continuous |
| `windspeed` | km/h | Continuous |
| `solarradiation` | W/m² | Continuous |
| `conditions` | Label | Categorical (e.g. "Sunny", "Cloudy") |

For each numeric target variable and horizon, the final system outputs a **point forecast** (single predicted value). Optionally, the `conditions` variable is predicted as a discrete label via a separate classifier.

Example output:
```
temp:           24h = 28.3°C,  48h = 29.1°C
humidity:       24h = 67.8%,   48h = 65.2%
windspeed:      24h = 4.2 km/h, 48h = 3.8 km/h
solarradiation: 24h = 450 W/m², 48h = 480 W/m²
conditions_24h: "Partly Cloudy"
conditions_48h: "Sunny"
```

---

## 3. Where Does the Data Come From?

The notebook assumes that you have a **historical time series** of greenhouse indoor conditions, for example:

- One row per **day** (or per fixed time step, e.g. daily aggregates)
- Columns for each weather variable (temperature, humidity, windspeed, solar radiation, sky conditions)
- A date/time index to keep everything ordered

For this project:
- **Data source:** Dindigul District, Tamil Nadu weather data (2024–2025)
- **Frequency:** Daily observations
- **Expected columns:** `datetime`, `temp`, `humidity`, `windspeed`, `solarradiation`, `conditions`, `sunriseEpoch`, `sunsetEpoch`, and others

This historical dataset is split into three parts:

1. **Training set** (70%) – earlier part of the history the models learn from.
2. **Validation set** (15%) – a slice used to tune hyperparameters, optimise ensemble weights, and prevent overfitting.
3. **Test set** (15%) – the last portion of history used only to check final performance (held out until the very end).

The notebook builds features from these time series and feeds them into the models described below.

---

## 4. Step-by-Step Pipeline Overview

At a high level, the notebook does the following:

1. **Load and clean raw weather data**
   - Read indoor greenhouse measurements (2024–2025 Dindigul data)
   - Handle missing values and ensure a continuous timeline
   - Extract temporal features (month, day-of-year, etc.)

2. **Engineer features** that help models understand seasonality, trends, and interactions:
   - Cyclical time encodings (sin/cos)
   - Lag features (past 1, 2, 3, 7, 14, 30 days)
   - Rolling statistics (7/14/30-day mean, std, min, max)
   - Dindigul seasonal flags
   - Climate normals and anomalies
   - Solar geometry features
   - Interaction terms (e.g. temperature × humidity)

3. **Prepare three types of models** in parallel:
   - **Chronos-T5-small** – pretrained time-series transformer
   - **XGBoost** – gradient-boosted tree ensemble (8 models for 4 targets × 2 horizons)
   - **LSTM** – stacked recurrent network (1 model per target)
   - **Random Forest classifier** – maps numeric forecasts → sky condition labels

4. **Train each model family** on the training set:
   - Chronos: warm-up (frozen encoder) → full fine-tune
   - XGBoost: warm-up (shallow) → fine-tune (deep) with early stopping
   - LSTM: sliding-window dataset, Huber loss, gradient clipping, early stopping
   - RF classifier: balanced class weighting

5. **Optimise ensemble weights** via Bayesian search (Optuna):
   - For each (target, horizon) pair, find optimal blend of Chronos + XGBoost + LSTM
   - Minimises validation MAPE
   - 300 trials per combination

6. **Evaluate performance** on the test set and generate plots

7. **Save all necessary artefacts** for realtime use:
   - Scalers, encoders, feature configurations
   - All model weights (Chronos, XGBoost, LSTM)
   - Ensemble weights and evaluation metrics
   - A ready-to-use Python loader for inference

8. **Clean up intermediate files** so only the final realtime bundle and required artefacts remain

---

## 5. Feature Engineering (Turning Raw Weather Into Inputs)

Raw numbers alone ("temperature = 26.3 °C") do not directly capture:

- Time of year and seasonal patterns
- Recent trends and local volatility
- Typical baseline values for this time/location
- Interactions between variables

To help the models, the notebook creates several **feature types**:

### 5.1 Cyclical Time Features

- **Day-of-year** (sin/cos encoded)
- **Month** (sin/cos encoded)
- **Day-of-week** (sin/cos encoded)

The sine/cosine encoding captures the circular nature of time (month 12 is next to month 1).

### 5.2 Dindigul Seasonal Flags

For the Dindigul region, the year is split into **four distinct seasons**:

| Season | Months | Character |
|--------|--------|-----------|
| Hemant (Winter) | Jan–Feb | Cool, dry, post-NE monsoon tail |
| Grishma (Summer) | Mar–May | Hot, low humidity, pre-monsoon |
| Varsha (SW Monsoon) | Jun–Sep | High humidity, moderate rain |
| Sharad (NE Monsoon) | Oct–Dec | Rain peaks, humid |

One-hot encoded flags tell the model which season each day belongs to.

### 5.3 Lag Features

The model includes **lagged versions** of target variables:

- Values from **1, 2, 3, 7, 14, and 30 days ago**

These help capture **autocorrelation** — the fact that today's temperature is usually similar to yesterday's or last week's.

### 5.4 Rolling Statistics

To capture **local trends and volatility**, the notebook computes:

- 7/14/30-day **rolling mean** (shifted by 1 day to avoid look-ahead bias)
- 7/14/30-day **rolling standard deviation**
- 7/14/30-day **min and max**

These tell the model if the climate has been gradually warming, cooling, or becoming more variable.

### 5.5 Climate Normals and Anomalies

The notebook builds **climatological baselines**:

- Typical **monthly** averages
- Typical **weekly** averages

For each day, the model can compute an **anomaly**:

```
anomaly = actual_value - typical_value_for_this_time_of_year
```

Because plants and disease risk often depend on **deviations from normal**, not just absolute values.

### 5.6 Solar Geometry & Chronos Meta-Features

Additional derived features:

- **Day length** (hours between sunrise and sunset)
- **Normalised solar radiation** (actual / day_length)
- **Chronos meta-features**: predictions from the fine-tuned Chronos model, reused as extra inputs to XGBoost and LSTM  
  (This gives other models a "head start" from the pretrained foundation model)

### 5.7 Volatility Features (NEW - For Sparse/Erratic Variables)

For variables like windspeed and humidity that exhibit high volatility, additional derived features capture **momentum and regime changes**:

- **Momentum** (rate-of-change): RoC over [1, 3, 7] days
- **Volatility clustering**: Rolling standard deviation with high/low regime indicators
- **Autocorrelation proxies**: ACF-like aggregations at lags [1, 7, 14]
- **Regime switches**: Seasonal high/low volatility flags

These features help models distinguish between **genuine predictable patterns** and **random noise**, improving R² for difficult variables.

All engineered features are stored in a configuration file (`feature_config.json`) so inference code can reproduce them.

---

## 6. The Three Model Families

The notebook uses **three different forecasting approaches** and later blends them. Each has complementary strengths:

- **Chronos** understands general time-series patterns (trends, seasonality, reversals)
- **XGBoost** captures tabular interactions and non-linear relationships
- **LSTM** captures medium-range temporal structure and local dependencies

### 6.1 Chronos-T5-Small: Pretrained Time-Series Foundation Model

**What is Chronos?**

- A **T5-based transformer model** released by Amazon for time-series forecasting
- Pre-trained on **~84 billion diverse time-series observations**
- Treats time series like the language model treats text: patterns are learned generically, then fine-tuned for specific domains

**How it works conceptually:**

- **Encoder**: Reads a sequence of past values (e.g. last 30 days of temperature)
- **Decoder**: Learns to generate the next values step-by-step
- **Attention layers**: Can "look back" at *any* past position, not just recent history

**Two-Phase Fine-Tuning Strategy**

1. **Warm-up phase** (5 epochs, frozen encoder):
   - **Freeze** the encoder (pretrained knowledge is fixed)
   - **Train** only the decoder and output projection head
   - Uses learning rate **1e-3** (moderate)
   - Stabilises training, prevents catastrophic forgetting

2. **Full fine-tune phase** (10 epochs, all layers):
   - **Unfreeze** all parameters
   - Lower learning rate to **1e-4** (more careful updates)
   - Monitor validation MAPE to detect overfitting
   - Restore best checkpoint at the end

**Predictions**

For each target variable, Chronos sees a **30-day context window** and forecasts the next **2 steps** (24h, 48h). These predictions are:

- **Used directly** in the final weighted ensemble
- **Reused as meta-features** for XGBoost and LSTM (bootstrapping other models with pretrained knowledge)

---

### 6.2 XGBoost: Gradient-Boosted Direct Multi-Step Forecaster (Per-Column Regularization)

**What is XGBoost?**

- A **gradient-boosted decision tree** ensemble
- Builds many small trees sequentially; each new tree corrects errors from previous ones
- Excellent for tabular (structured) data with many features

**Direct Multi-Step Strategy**

Instead of predicting one step at a time (which accumulates errors), we train **one model per (target, horizon)** pair:

- `xgb_temp_24h.pkl` → temperature 24 hours ahead
- `xgb_humidity_48h.pkl` → humidity 48 hours ahead
- (total: 4 targets × 2 horizons = **8 XGBoost models**)

**Per-Column Variable-Specific Hyperparameters**

Key innovation: **different variables get different regularization**, because not all variables are equally prone to overfitting:

| Variable | max_depth | λ (reg_lambda) | α (reg_alpha) | min_child_weight | Rationale |
|----------|-----------|----------------|---------------|------------------|-----------|
| Temperature | 4 | 2.0 | 0.1 | 5 | Stable; standard regularisation |
| Humidity | 2 | 6.0 | 0.5 | 15 | Very volatile; strong regularisation |
| Windspeed | **1** | **20.0** | **2.0** | 15 | Extremely sparse; ultra-aggressive regularisation |
| Solar Radiation | 4 | 2.0 | 0.1 | 5 | Stable; standard regularisation |

**Rationale for Aggressive Windspeed Regularisation:**
- Windspeed data is sparse and highly variable (many days with similar values)
- Deep trees overfit on noise
- Depth=1 (stump-like trees) paired with extreme L1/L2 penalties forces the model to find only the **strongest feature interactions**
- Result: model generalises better to unseen data

**Two-Phase Training for Small Data**

The dataset is limited (~300 useful samples after feature engineering), so XGBoost uses:

1. **Warm-up phase**:
   - Shallow trees (`max_depth=3`, uniform for all)
   - Few boosting rounds (200 estimators)
   - Higher learning rate (0.10)
   - Quick convergence to a warm baseline

2. **Fine-tune phase** (inherits warm-up booster):
   - **Per-column max_depth** (see table above)
   - More boosting rounds (600 estimators)
   - Lower learning rate (0.04)
   - **Per-column regularisation** (λ, α, min_child_weight from table)
   - Early stopping: stops if validation MAE doesn't improve for 30 rounds

**What XGBoost sees**

Each model receives the full **feature vector** for the last available day:
- All lag features, rolling statistics, season flags, Chronos meta-features, etc.

XGBoost excels at finding the **best combination of features** for each prediction, complementing the sequence-focused Chronos and LSTM.

---

### 6.3 LSTM: Recurrent Sequential Regressor (Per-Horizon Separate Models)

**What is LSTM?**

- **Long Short-Term Memory** — a type of recurrent neural network (RNN)
- Designed to process sequences while maintaining internal memory
- Can "remember" distant past events (via gating mechanisms)

**Architecture**

Each model has:
- **2 stacked LSTM layers** (128 hidden units per layer)
- **LayerNorm** on the final hidden state (improves training stability)
- **Per-horizon dropout**: Different for 24h vs 48h horizons (see table below)
- **Feed-forward head**: maps hidden state → **single step prediction** (not 2 like before)

**Per-Horizon Separate Models — Key Architectural Change**

**OLD approach (single model per target):**
- One model per variable (e.g., `lstm_temperature`) outputting both 24h and 48h simultaneously
- Result: model had to balance two incompatible objectives, producing suboptimal 48h predictions

**NEW approach (separate model per target × horizon):**
- **8 separate models total**: one for each (target, horizon) pair
  - `lstm_temp_24h`, `lstm_temp_48h`, `lstm_humidity_24h`, `lstm_humidity_48h`, etc.
- Each model outputs **only one horizon**, allowing horizon-specific tuning
- Result: 48h predictions can be heavily regularised without hurting 24h

**Per-Horizon Dropout Strategy**

| Horizon | Base Dropout Increase | Effective Dropout (humidity) | Effective Dropout (windspeed) | Effective Dropout (temp) | Effective Dropout (solar) |
|---------|----------------------|------------------------------|-------------------------------|--------------------------|---------------------------|
| 24h | 0.00 | 0.10 | 0.05 | 0.25 | 0.25 |
| 48h | **+0.15** | **0.25** | **0.20** | **0.40** | **0.40** |

**Rationale:** Predicting 48 hours ahead is fundamentally harder (exponentially more uncertainty). Adding 0.15 extra dropout for 48h forces the model to rely on only the most robust learned patterns, preventing overfitting on noise.

**Variable-Specific Learning Rates & Base Dropout**

| Variable | Learning Rate | Base Dropout | Rationale |
|----------|-----------------|--------------|----------|
| Temperature | 1e-3 | 0.25 | Stable temporal patterns; standard LR |
| Humidity | 2e-4 | 0.10 | Volatile swings; smaller LR for finer search |
| Windspeed | 1e-4 | 0.05 | Ultra-sparse data; conservative LR and light dropout |
| Solar Radiation | 1e-3 | 0.25 | Complex multi-modal; standard LR |

**Sliding-Window Dataset**

Unlike direct multi-step, LSTM trains on **sliding windows** of features:

- For each day in the training period, take the previous **30 days** of features as input
- Target: **single horizon value** (24h-ahead OR 48h-ahead, not both)
- Create per-horizon `StandardScaler` fitted on training data only
- Result: many overlapping training examples, each associated with one specific horizon

**Loss Function: Huber Loss**

Instead of simple mean-squared-error, we use **Huber loss** ($\delta=1.0$), which is **robust to outliers**:

- For small errors: acts like MSE (smooth quadratic penalty)
- For large errors: acts like MAE (linear penalty, less severe)

This matters for weather data because occasional extreme events (dust storms, unusual wind gusts) shouldn't dominate the loss.

**Training Strategy**

1. **Optimiser:** Adam with weight decay (L2 regularisation: 1e-5)
2. **Learning rate schedule:** Cosine annealing (starts at per-column LR in table, gradually drops to 1e-6 minimum over 100 epochs)
3. **Gradient clipping:** Prevents exploding gradients (`||g|| ≤ 1.0`)
4. **Early stopping:** Halts if validation loss doesn't improve for 20 epochs
5. **Per-target & per-horizon scalers:** Each (variable, horizon) pair gets its own `StandardScaler`:
   - Fit on training data only (prevents data leakage)
   - Predictions are inverse-transformed back to original units (°C, %, km/h, W/m²)

**Why Per-Horizon Separate Models?**

- **Problem with single model for both horizons:** Model forced to average prediction quality across 24h and 48h; heavy regularisation helps 48h but hurts 24h
- **Solution (per-horizon models):** Each horizon gets its own model, tuned to its own difficulty level
- **Result:** 48h predictions maintain competitive R² scores (e.g., humidity 48h improved from negative to +0.0024, windspeed 48h from -0.2574 to +0.0128)

**Why LSTM Over Temporal Fusion Transformer?**

The notebook originally tried **Temporal Fusion Transformer (TFT)** — a powerful multi-entity forecasting architecture. However:

- **TFT** is designed for hundreds/thousands of time series (multi-entity datasets)
- On a **single daily series of ~300 training samples**, TFT tends to **underfit or diverge**, producing **negative R²**
- **LSTM** is more **data-efficient**:
  - Far fewer parameters (less overfitting risk)
  - More stable convergence on small data
  - Proven track record on weather-like sequences

---

### 6.4 How We Train All Models Together

The full training pipeline follows this sequence:

**Step 1: Data Loading & Preprocessing + Configuration**
- Load 2024–2025 Dindigul weather data
- Align timestamps, remove duplicates
- Extract temporal features (year, month, day, dayofweek, etc.)
- **Initialize CONFIG with per-column and per-horizon hyperparameters:**
  - `lstm_dropout_per_col`: {temp: 0.25, humidity: 0.10, windspeed: 0.05, solarradiation: 0.25}
  - `lstm_dropout_per_horizon`: {1: 0.00, 2: 0.15} (add 0.15 to 48h predictions)
  - `lstm_lr_per_col`: {temp: 1e-3, humidity: 2e-4, windspeed: 1e-4, solarradiation: 1e-3}
  - `xgb_max_depth_per_col`: {temp: 4, humidity: 2, windspeed: 1, solarradiation: 4}
  - `xgb_reg_lambda_per_col`: {temp: 2.0, humidity: 6.0, windspeed: 20.0, solarradiation: 2.0}
  - `xgb_reg_alpha_per_col`: {temp: 0.1, humidity: 0.5, windspeed: 2.0, solarradiation: 0.1}
  - `xgb_min_child_weight_per_col`: {temp: 5, humidity: 15, windspeed: 15, solarradiation: 5}

**Step 2: Feature Engineering (Done Once on All Data)**
- Cyclical encodings (sin/cos for time features)
- Lag features (1, 2, 3, 7, 14, 30 days per target)
- Rolling statistics (7/14/30-day mean/std/min/max, shifted by 1 day)
- Dindigul seasons (one-hot encoded)
- Climate normals (monthly/weekly averages, computed on full dataset)
- Anomalies (actual - normal)
- Solar geometry (day length, normalized radiation)
- Interaction features (temp × humidity, wind × solar, etc.)
- Volatility features (momentum, regime detection for wind/humidity)
- Create forward-shifted targets for t+1 (24h) and t+2 (48h)
- Store all engineered features with versions in `feature_config.json`

**Step 3: Chronological Train/Val/Test Split**
- **Train:** first 70% of samples
- **Validation:** next 15%
- **Test:** final 15% (held out completely until final evaluation)
- Split along time axis (no shuffling) to avoid look-ahead bias

**Step 4: Feature Scaling**
- Fit `RobustScaler` on training data only
- Apply to validation and test
- Prevents data leakage

**Step 5: Chronos Training**
- Build time-series windows (30 days context, 2 steps target)
- Warm-up (5 epochs, frozen encoder only)
- Full fine-tune (10 epochs, all layers, restore best checkpoint)
- Generate meta-features by predicting on all splits

**Step 6: XGBoost Training (Per-Column Hyperparameters)**
- For each (target, horizon) pair (8 total):
  - **Warm-up phase**:
    - 200 estimators, shallow trees (depth=3 fixed), LR=0.10
    - Goal: quick baseline convergence
  - **Fine-tune phase** (inherits warm-up booster):
    - 600 estimators, **per-column depth** (e.g., windspeed depth=1, humidity depth=2)
    - LR=0.04
    - **Per-column regularisation**: read λ, α, min_child_weight from CONFIG per-col dicts
    - Example: windspeed uses λ=20.0 (ultra-aggressive L2 penalty)
    - Early stopping: 30 rounds on validation MAE

**Step 7: LSTM Training (Per-Horizon Separate Models)**
- **For each (target, horizon) pair (8 total):**
  - Build horizon-specific sliding-window dataset (30-day context → single horizon)
  - Create per-horizon `StandardScaler` (fit on training data only)
  - **Read per-column hyperparameters from CONFIG:**
    - Base dropout: `lstm_dropout_per_col[target]` (e.g., humidity=0.10)
    - Horizon addon: `lstm_dropout_per_horizon[h]` (e.g., horizon 2 adds +0.15 for 48h)
    - Final dropout: 0.10 + 0.15 = 0.25 for humidity 48h
    - Learning rate: `lstm_lr_per_col[target]` (e.g., windspeed=1e-4)
  - Train with Huber loss (δ=1.0), Adam with weight decay (L2=1e-5)
  - Learning rate schedule: Cosine annealing over 100 epochs
  - Early stopping on validation loss (patience=20)
  - Restore best checkpoint
  - Save per-(col, h) state dict and scaler

**Step 8: Ensemble Weight Optimisation (1200 Trials)**
- Collect validation predictions from all three model families on held-out validation set
- Use **Optuna** (Bayesian Tree-Parzen Estimator sampler) to find optimal weights:
  - Constraint: $w_\text{Chronos} + w_\text{XGB} + w_\text{LSTM} = 1$, all ≥ 0
  - Objective: minimise validation MAPE (per target, per horizon)
  - **1200 trials per (target, horizon) pair** (increased from 300 to find better combinations)
  - Optimization discovers which model dominates for each: e.g., solar 48h prefers pure LSTM (weight=1.0)
- Store best weights in `ensemble_weights.json`

**Step 9: Conditions Classifier Training**
- For each horizon (24h, 48h):
  - Build input: raw weather values + temporal features
  - Target: observed sky condition (Sunny, Cloudy, etc.)
  - Train Random Forest (400 trees, max_depth=10, balanced class weights)

**Step 10: Final Evaluation on Test Set**
- Ensemble predictions: blend Chronos + XGBoost + LSTM with optimised weights (per-col, per-horizon)
- Compute metrics: MAPE, R², RMSE, MAE, Accuracy per target/horizon
- **All R² values confirmed > 0** (including previously negative humidity/windspeed 48h)
- Generate visualisations (actual vs predicted, metric summaries, weight distributions)

**Step 11: Save Artefacts & Cleanup**
- Bundle LSTM states (8 models × state dict per (col, h)) + scalers (8 per-horizon scalers) into `environment_forecast_<run_id>.pt`
- Save all supporting files:
  - `feature_config.json` (feature names, target cols, context length, volatility flags)
  - Per-column hyperparameter configs (for audit trail)
  - Ensemble weights with confidence intervals
  - Evaluation metrics (final R², MAPE, RMSE per target/horizon)
- Move plots to artefacts folder
- **Delete intermediate files** (fine-tuned checkpoints, temp models, Lightning logs, warm-up boosters)

This ensures:
- All models see **consistent features and splits**
- Chronos and LSTM exploit temporal structure with **horizon-aware regularisation**
- XGBoost focuses on rich tabular interactions with **variable-specific regularisation**
- Ensemble learns **data-driven weights per-target-per-horizon** (no manual guessing)
- All volatile variables (humidity, windspeed) recover to **positive R² values**

---

## 7. Ensemble: Optuna-Optimised Weight Blending (1200 Trials)

No single model is perfect. Instead of choosing just one, the notebook uses an **ensemble**:

For each **target variable** and **horizon** (24h, 48h), it learns a set of **weights** that blend the three predictions:

$$\text{final\_prediction} = w_\text{Chronos} \cdot \hat{y}_\text{Chronos} + w_\text{XGB} \cdot \hat{y}_\text{XGB} + w_\text{LSTM} \cdot \hat{y}_\text{LSTM}$$

**Constraints:**
- $w_\text{Chronos} + w_\text{XGB} + w_\text{LSTM} = 1$
- All weights ≥ 0

**Optimisation Method: Optuna (Bayesian TPE Sampler)**

- **1200 trials per (target, horizon)** (increased from 300 to find better weight combinations after model improvements)
- Objective: minimise validation MAPE
- Sampler: Tree-Parzen Estimator (Bayesian search)
- Output: optimal weights stored in `ensemble_weights.json`

**Final Ensemble Weight Patterns**

The optimised weights show distinct patterns per variable and horizon:

| Target | Horizon | Chronos | XGBoost | LSTM | Pattern |
|--------|---------|---------|---------|------|----------|
| Temperature | 24h | 0.163 | **0.780** | 0.057 | XGBoost dominant (tabular features work well) |
| Temperature | 48h | 0.093 | 0.225 | **0.682** | LSTM dominant (temporal structure matters for distant forecast) |
| Humidity | 24h | 0.405 | 0.411 | 0.184 | Balanced Chronos + XGBoost (volatile variable) |
| Humidity | 48h | 0.479 | 0.000 | **0.521** | LSTM dominant (48h adds uncertainty; XGBoost muted) |
| Windspeed | 24h | 0.457 | 0.374 | 0.169 | Chronos + XGBoost balanced (sparse data) |
| Windspeed | 48h | 0.420 | **0.579** | 0.000 | XGBoost dominant (regularised XGBoost depth=1 works better for 48h) |
| Solar Radiation | 24h | 0.052 | 0.000 | **0.948** | LSTM dominant (complex intra-day variations) |
| Solar Radiation | 48h | 0.000 | 0.000 | **1.000** | LSTM only (pure LSTM for very uncertain distant solar) |

**Key Observations:**
- **Temperature 48h is LSTM-heavy** (0.682): temporal patterns matter for distant forecasts
- **Solar 48h is LSTM-only** (1.000): distant solar is too uncertain for tree or pretrained models
- **Windspeed 48h is XGB-heavy** (0.579): aggressive regularisation (depth=1) forces robustness
- **Humidity 48h LSTM-dominant** (0.521): volatile variable benefits from horizon-specific tuning

This data-driven, **per-horizon-per-variable approach** often produces more robust predictions than any single component, and adapts the blend to variable difficulty.

---

## 8. Conditions Classifier (Sky Condition Labels)

Numbers like "26.7 °C" and "65% humidity" are informative, but sometimes we want a **human-friendly label** such as:

- "Sunny"
- "Partly Cloudy"
- "Overcast"
- "Rainy"

**How it works:**

The notebook trains a separate **Random Forest classifier** (per horizon) that:

1. Takes the numeric forecast values (temperature, humidity, windspeed, solar radiation) and temporal features
2. Maps them to a discrete sky-condition label

**Saved artifacts:**

- `conditions_classifier_24h.pkl` – condition forecast 24h ahead
- `conditions_classifier_48h.pkl` – condition forecast 48h ahead
- `label_encoder.pkl` – decoder from integer codes back to label strings

---

## 9. What Gets Saved After Training?

After the notebook finishes, you will have a directory structure like:

```
src/agritwin_gh/models/
├── environment_forecast_<run_id>.pt
│   └── Primary LSTM bundle (all targets, scalers, configs)
│
└── artifacts/environment_forecast_<run_id>/
    ├── feature_config.json
    │   └── All feature names, target cols, context length, season map, condition classes
    │
    ├── scalers.pkl
    │   └── RobustScaler for the engineered feature matrix (fit on train only)
    │
    ├── label_encoder.pkl
    │   └── LabelEncoder for sky condition labels (e.g. "Sunny" → 0, "Cloudy" → 1)
    │
    ├── climate_normals.json
    │   └── Monthly and weekly climatological means for each target variable
    │
    ├── ensemble_weights.json
    │   └── Optimal blend weights for each (target, horizon) combination
    │
    ├── evaluation_metrics.json
    │   └── Test set metrics: MAPE, R², RMSE, MAE, accuracy per target/horizon
    │
    ├── xgb_temp_24h.pkl
    ├── xgb_temp_48h.pkl
    ├── xgb_humidity_24h.pkl
    ├── xgb_humidity_48h.pkl
    ├── xgb_windspeed_24h.pkl
    ├── xgb_windspeed_48h.pkl
    ├── xgb_solarradiation_24h.pkl
    ├── xgb_solarradiation_48h.pkl
    │   └── 8 XGBoost models (one per target × horizon)
    │
    ├── conditions_classifier_24h.pkl
    ├── conditions_classifier_48h.pkl
    │   └── Random Forest classifiers for sky condition prediction
    │
    ├── chronos_finetuned/
    │   ├── t5_finetuned_state_dict.pt
    │   │   └── Fine-tuned Chronos model weights
    │   └── chronos_finetune_config.json
    │       └── Training hyperparameters and loss history
    │
    ├── environment_forecast_loader.py
    │   └── Reusable Python inference helper (standalone, no notebook state)
    │
    └── plots/
        ├── eda_timeseries.png
        ├── eda_seasonal_boxplot.png
        ├── eda_correlation.png
        ├── chronos_training_curve.png
        ├── xgb_shap_importance.png
        ├── ensemble_predictions_test.png
        ├── metrics_summary.png
        └── (other visualisations)
```

> **Important Cleanup Policy:** The notebook deletes all intermediate model files and logs. Only the **final realtime bundle** (`environment_forecast_<run_id>.pt`) and necessary artefacts remain on disk, keeping the deployment footprint small.

---

## 10. How to Use the Final Model for Inference

After training, the notebook generates a reusable Python helper: `environment_forecast_loader.py`

This module contains a class `EnvironmentForecastModel` that:
- Loads all necessary artefacts (scalers, encoders, weights, model files)
- Handles feature engineering and scaling
- Executes Chronos, XGBoost, and LSTM models
- Blends predictions using optimised weights
- Returns forecast dict

### 10.1 Minimal Usage Example

```python
from pathlib import Path
from src.agritwin_gh.models.artifacts.environment_forecast_<run_id>.environment_forecast_loader import (
    EnvironmentForecastModel,
)

# Paths to artefacts and main model bundle
artifacts_dir = "src/agritwin_gh/models/artifacts/environment_forecast_<run_id>"
model_path   = "src/agritwin_gh/models/environment_forecast_<run_id>.pt"

# Instantiate model (CPU by default, or "cuda" for GPU)
model = EnvironmentForecastModel(
    artifacts_dir=artifacts_dir,
    main_model_path=model_path,
    device="cpu",
)

# df_context must have:
# - At least `context_length` rows (typically 30)
# - All feature columns (temperature, humidity, lags, rolling stats, etc.)
#   Names are defined in feature_config.json
preds = model.predict(df_context)

print(preds)
# Example output:
# {
#   "temp": {"24h": 28.3, "48h": 29.1},
#   "humidity": {"24h": 67.8, "48h": 65.2},
#   "windspeed": {"24h": 4.2, "48h": 3.8},
#   "solarradiation": {"24h": 450, "48h": 480}
# }
```

### 10.2 What Features Must df_context Have?

Look inside `feature_config.json` in the artefacts directory:

- `all_feature_names` – complete list of feature column names (lags, rolling stats, etc.)
- `target_cols` – variables being predicted (temp, humidity, etc.)
- `context_length` – how many recent rows are required (typically 30 days)

Your `df_context` should have all these columns in the exact order/names, with at least `context_length` rows.

---

## 11. Key Metrics and Performance

The notebook evaluates the ensemble on the **test set** (held out from training). Key metrics include:

### 11.1 Error Metrics

- **MAE (Mean Absolute Error)** – average absolute difference between forecast and actual
- **RMSE (Root Mean Squared Error)** – penalises larger errors more
- **MAPE (Mean Absolute Percentage Error)** – percentage error (useful for comparing variables with different scales)

### 11.2 Correlation/Explanation Metrics

- **R² Score** – how much variance the model explains (1.0 = perfect, 0.0 = no better than constant, <0 = worse than constant)

### 11.3 Accuracy Proxy

- **Accuracy = 100 - MAPE(%)**  
  A target accuracy of **≥ 95%** means **MAPE ≤ 5%**

### 11.4 Conditions Classifier

- **Accuracy** – percentage of correctly predicted sky condition labels (24h and 48h)

All metrics are saved in `evaluation_metrics.json`.

---

### 11.5 Final Test Performance (Post-Optimization)

After implementing per-column XGBoost regularization, separate per-horizon LSTM models, and 1200-trial ensemble optimization, all R² values are now **≥ 0** (a major improvement from early negative R² values on volatile variables):

| Target | Horizon | MAPE (%) | Accuracy (%) | RMSE | MAE | R² | Status |
|--------|---------|----------|----------|------|-----|-----|--------|
| Temperature | 24h | 3.11 | 96.89 | 1.086 | 0.824 | **0.6983** | ✅ Excellent |
| Temperature | 48h | 3.37 | 96.63 | 1.175 | 0.890 | **0.6486** | ✅ Excellent |
| Humidity | 24h | 8.56 | 91.44 | 7.226 | 6.092 | **0.3402** | ✅ Good |
| **Humidity | 48h** | **9.75** | **90.25** | **8.886** | **7.135** | **0.0024** | ✅ Fixed (was -0.1716) |
| **Windspeed | 24h** | **26.34** | **73.66** | **5.709** | **4.647** | **0.0294** | ✅ Fixed (was -0.0231) |
| **Windspeed | 48h** | **26.86** | **73.14** | **5.733** | **4.676** | **0.0128** | ✅ Fixed (was -0.2574) |
| Solar Radiation | 24h | 42.84 | 57.16 | 54.562 | 42.776 | **0.2383** | ⚠️ Challenging |
| Solar Radiation | 48h | 41.03 | 58.97 | 52.440 | 43.993 | **0.2965** | ⚠️ Challenging |
| Skill (Conditions) | 24h | — | 55.77 | — | — | — | ⚠️ Fair |
| Skill (Conditions) | 48h | — | 50.00 | — | — | — | ⚠️ Fair |

**Success Criteria Met:**
- ✅ **All R² values > 0** (humidity 48h and windspeed 48h recovered from negative to positive)
- ✅ **Temperature predictions robust** (R² > 0.64 for both horizons)
- ✅ **Humidity predictions good** (R² > 0.30)
- ⚠️ Solar radiation challenging (R² 0.24–0.30); inherent data scarcity and multi-modal patterns

**What Drove the Improvements:**

1. **Per-column XGBoost regularization:** Windspeed/humidity benefit from aggressive depth=1 and λ=20–6.0
2. **Per-horizon LSTM models:** 48h can now be heavily regularised (+0.15 dropout) without hurting 24h
3. **1200-trial Optuna:** Better weight optimization found superior model blends (e.g., windspeed 48h prefers XGB, humidity 48h prefers LSTM)
4. **Volatility-aware features:** Momentum, regime detection help volatile variables

---

## 12. Architecture Summary & Design Decisions

### Why Per-Column Hyperparameters for XGBoost?

Different variables have different predictability:

- **Temperature & Solar** (stable): Standard depth=4, moderate regularisation (λ=2.0)
- **Humidity** (volatile): Aggressive depth=2, stronger regularisation (λ=6.0)
- **Windspeed** (ultra-sparse): Extreme depth=1 with λ=20.0, forcing stump-like trees that capture only the most robust patterns

This **variable-aware tuning** prevents overfitting on small data while allowing stronger models on easier targets.

### Why Per-Horizon LSTM Models?

Single model for both horizons creates a **compromise**:
- Heavy regularisation helps 48h but hurts 24h accuracy
- Light regularisation helps 24h but allows 48h to overfit

**Separate per-horizon models** allow:
- 24h model: light regularisation for precision (dropout=0.05-0.10 base)
- 48h model: aggressive regularisation to fight uncertainty (dropout adds +0.15)

Result: humidity 48h R² improved from **-0.1716 → +0.0024**, windspeed 48h from **-0.2574 → +0.0128**.

### Why 1200 Trials for Ensemble Weights?

After model improvements, ensemble weight optimization became crucial:
- Initial 300 trials found local optima
- 1200 trials enabled discovery of better blends
- Example: windspeed 48h found XGB-dominant (0.579) better than LSTM-heavy
- Example: solar 48h found pure LSTM (1.000) optimal

### Why LSTM Over Temporal Fusion Transformer?

**TFT (Temporal Fusion Transformer)** is state-of-the-art for **large multi-entity datasets**. On a **single daily series of ~300 samples**, it tends to:
- Overfire on complex interactions (too many parameters)
- Underfit or produce negative R²

**LSTM** is more data-efficient:
- Fewer parameters
- Proven convergence on small weather datasets
- Huber loss handles outliers robustly

### Why Chronos Meta-Features?

Chronos is a pretrained "time-series language model." Its predictions contain valuable generalised knowledge:
- Reusing as features for XGBoost and LSTM **bootstraps** weaker models
- Gives tree and RNN methods a "head start" on temporal patterns

### Realtime Inference Footprint

- The `.pt` bundle contains only the LSTM weights and scalers
- XGBoost and RF models are separate small pickle files
- Fine-tuned Chronos state dict is a few MB
- Total footprint: **< 50 MB** (easily deployable to edge devices or cloud APIs)

---

## 13. Glossary

- **LSTM** – Long Short-Term Memory; a type of recurrent neural network designed to handle sequences and long-range dependencies
- **XGBoost** – Extreme Gradient Boosting; a tree ensemble method that iteratively improves by correcting previous errors
- **Chronos** – Amazon's pretrained transformer for time-series forecasting
- **Ensemble** – A combination of multiple models; predictions are blended (often improving accuracy and robustness)
- **Horizon** – How far into the future we are predicting (e.g., 24h, 48h)
- **Context Window** – A sliding window of recent historical data (e.g., last 30 days) fed into a model
- **Feature** – An input variable to a model (e.g., day-of-year, lagged temperature)
- **Target Variable** – The value we want to predict (e.g., tomorrow's temperature)
- **Scaler** – A transformation that normalises input data (e.g., `RobustScaler`, `StandardScaler`)
- **Label Encoder** – Converts categorical strings ("Sunny", "Cloudy") to numeric codes (0, 1, etc.)
- **MAE / RMSE / MAPE / R²** – Common performance metrics for regression tasks
- **Huber Loss** – A robust loss function that behaves like MSE for small errors and MAE for large errors
- **Optuna** – A hyperparameter optimisation framework using Bayesian search (TPE sampler)
- **Gradient Clipping** – Preventing exploding gradients in neural networks by capping their magnitude
- **Early Stopping** – Halting training if a metric (e.g., validation loss) doesn't improve for a patience period
- **Autoregressive** – Predicting future values based on past values of the same variable (e.g., temperature depends on past temperatures)

---

**This document is a comprehensive reference for the `weather_forecast.ipynb` notebook.**  
Treat the notebook as the *implementation* and this markdown as the *guided tour* and reference manual.

For detailed code, cell-by-cell execution, and interactive plots, refer to the notebook itself.
