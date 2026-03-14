# Greenhouse Weather Forecast Model (Chronos + XGBoost + LSTM)

> **Who is this for?**  
> Anyone — farmer, student, developer, or complete beginner — who wants to understand *what* the `weather_forecast.ipynb` notebook does, *why* it matters, and *how* to use the final model in practice. No maths or machine-learning background is required.

---

## Table of Contents

1. [Big Picture: What Problem Are We Solving?](#1-big-picture-what-problem-are-we-solving)
2. [What Exactly Does the Model Predict?](#2-what-exactly-does-the-model-predict)
3. [Where Does the Data Come From?](#3-where-does-the-data-come-from)
4. [Step-by-Step Pipeline Overview](#4-step-by-step-pipeline-overview)
  - [4.1 Training & Inference Flow Diagram](#41-training--inference-flow-diagram)
5. [Feature Engineering (Turning Raw Weather Into Inputs)](#5-feature-engineering-turning-raw-weather-into-inputs)
6. [The Three Model Families](#6-the-three-model-families)
   - [6.1 Chronos-T5 Time-Series Foundation Model](#61-chronos-t5-time-series-foundation-model)
   - [6.2 XGBoost Gradient-Boosted Trees](#62-xgboost-gradient-boosted-trees)
   - [6.3 LSTM Neural Network](#63-lstm-neural-network)
  - [6.4 How We Train All Models Together](#64-how-we-train-all-models-together)
7. [Ensemble: Blending the Three Models](#7-ensemble-blending-the-three-models)
8. [Conditions Classifier (Sky Condition Labels)](#8-conditions-classifier-sky-condition-labels)
9. [What Gets Saved After Training?](#9-what-gets-saved-after-training)
10. [How to Use the Final Model for Inference](#10-how-to-use-the-final-model-for-inference)
11. [Realtime Test Script](#11-realtime-test-script)
12. [Understanding the Metrics and Plots](#12-understanding-the-metrics-and-plots)
13. [Typical Realtime Scenario Walkthrough](#13-typical-realtime-scenario-walkthrough)
14. [Common Questions (FAQ)](#14-common-questions-faq)
15. [Glossary](#15-glossary)

---

## 1. Big Picture: What Problem Are We Solving?

Inside a controlled greenhouse, **weather is not just outside** — it is also **inside**:

- Air temperature
- Relative humidity
- Solar radiation / light intensity
- Possibly wind speed, pressure, and other variables

If we can **predict the next 24–48 hours of indoor conditions**, we can:

- Adjust heating, cooling, fans, and fogging *before* conditions drift out of the safe zone.
- Plan irrigation and nutrient dosing more precisely.
- Anticipate disease risk windows where temperature and humidity combinations are dangerous.
- Feed a **digital twin** (AgriTwin-GH) that simulates future plant growth and disease.

The `weather_forecast.ipynb` notebook builds a **multi-step time-series forecasting system** for the greenhouse environment. It does not just guess tomorrow’s value from thin air — it learns from months of historical sensor data.

---

## 2. What Exactly Does the Model Predict?

The model predicts **future values of several indoor climate variables** for two horizons:

- **24 hours ahead** ("24h")
- **48 hours ahead** ("48h")

Typical target variables include (exact names come from the dataset):

- Indoor temperature (°C)
- Indoor relative humidity (%)
- Possibly CO₂ concentration, VPD (vapour pressure deficit), or other derived variables

For each target variable, the final system outputs:

```text
<variable_name>: 24h = <predicted_value>, 48h = <predicted_value>
```

These predictions are later combined into *sky condition* labels such as "Sunny", "Cloudy", etc., via a separate classifier.

---

## 3. Where Does the Data Come From?

The notebook assumes that you have a **historical time series** of greenhouse indoor conditions, for example:

- One row per **day** (or per fixed time step, e.g. hourly/daily aggregates)
- Columns for each weather variable (temperature, humidity, etc.)
- A date or time index to keep everything ordered

This historical dataset is split into three parts:

1. **Training set** – earlier part of the history the models learn from.
2. **Validation set** – a slice used to tune hyperparameters and prevent overfitting.
3. **Test set** – the last portion of history used only to check final performance.

The notebook builds features from these time series and feeds them into the models described below.

---

## 4. Step-by-Step Pipeline Overview

At a high level, the notebook does the following:

1. **Load and clean raw weather data**
   - Read indoor greenhouse measurements.
   - Handle missing values and ensure a continuous timeline.

2. **Engineer features** that help models understand seasonality and trends.

3. **Prepare three types of models**:
   - Chronos (a pretrained time-series model).
   - XGBoost (tree-based models for each target and horizon).
   - LSTM (a neural network that sees sliding windows of recent days).

4. **Train each model family** on the training set.

5. **Combine models using an ensemble**: a weighted blend of Chronos + XGBoost + LSTM for each (target, horizon) pair.

6. **Train a conditions classifier** that converts numeric forecasts into discrete sky-condition labels.

7. **Evaluate performance** on the test set and generate plots.

8. **Save all necessary artefacts** for realtime use:
   - Feature configuration and scalers.
   - Model weights.
   - Ensemble weights and evaluation metrics.
   - A ready-to-use Python loader for inference.

9. **Clean up intermediate files** so only the final realtime model remains.

### 4.1 Training & Inference Flow Diagram

The diagram below shows the end-to-end flow from raw data to realtime predictions.

```mermaid
flowchart LR
  subgraph Offline_Training[Offline Training Phase]
    A[Raw greenhouse
    weather history] --> B[Data cleaning
    & alignment]
    B --> C[Feature engineering
    (lags, rolling stats,
    seasons, anomalies...)]

    C --> D1[Chronos training
    (warm-up + fine-tune)]
    C --> D2[XGBoost training
    per target & horizon]
    C --> D3[LSTM training
    with sliding windows]

    D1 --> E[Validation forecasts]
    D2 --> E
    D3 --> E
    E --> F[Optuna search for
    ensemble weights]

    F --> G[Artefact export]
    C --> G
  end

  subgraph Artefacts[Saved Artefacts]
    G --> H1[feature_config.json]
    G --> H2[scalers.pkl
    & label_encoder.pkl]
    G --> H3[ensemble_weights.json]
    G --> H4[xgb_*.pkl,
    conditions_classifier_*.pkl]
    G --> H5[chronos_finetuned/*]
    G --> H6[environment_forecast_<run_id>.pt
    (LSTM bundle)]
    G --> H7[environment_forecast_loader.py]
  end

  subgraph Realtime_Inference[Realtime Inference Phase]
    I[Latest context window
    from sensors/database] --> J[EnvironmentForecastModel.predict]
    H1 --> J
    H2 --> J
    H3 --> J
    H4 --> J
    H5 --> J
    H6 --> J
    H7 --> J
    J --> K[24h & 48h
    forecasts per variable]
    K --> L[Optional conditions
    classifier
    (Sunny/Cloudy/...)]
    L --> M[Digital twin,
    dashboards,
    control policies]
  end
```

---

## 5. Feature Engineering (Turning Raw Weather Into Inputs)

Raw numbers alone ("temperature = 26.3 °C") do not directly capture:

- Time of year
- Season
- Recent trends
- Typical patterns for this location

To help the models, the notebook creates several **feature types**:

### 5.1 Time-Based Features

- **Day-of-year** (1–365) and **month** (1–12)
- **Day-of-week** (1–7)

To capture circular nature (e.g. month 12 is next to month 1), these are often encoded as **cyclical features** using sine and cosine.

### 5.2 Dindigul Seasonal Flags

For the Dindigul region, the year is split into **four seasons**:

- Winter
- Summer
- South-West Monsoon
- North-East Monsoon

Binary flags indicate which season each day belongs to. This helps the model know if a particular day belongs to a hot, dry, or humid monsoon period.

### 5.3 Lag Features

The model includes **lagged versions** of target variables:

- Values from **1, 2, 3, 7, 14, and 30 days ago**

These help capture autocorrelation — the fact that today’s temperature is often similar to previous days.

### 5.4 Rolling Statistics

To capture local trends and volatility, the notebook computes rolling-window features like:

- 7/14/30-day **mean**
- 7/14/30-day **standard deviation**
- 7/14/30-day **min and max**

These tell the model if the climate has been gradually warming, cooling, or becoming more variable.

### 5.5 Climate Normals and Anomalies

The notebook also builds **climate normals**:

- Typical **monthly** averages
- Typical **weekly** averages

For each day, the model can compare the current value to the typical value for that time of year and compute an **anomaly**:

```text
anomaly = actual_value - typical_value_for_this_time_of_year
```

This is helpful because plants and disease risk often depend on **deviations from normal**, not just absolute values.

### 5.6 Solar Geometry & Meta-Features

Additional features may include:

- **Day length** (hours of daylight)
- **Normalised solar radiation**
- **Chronos meta-features**: forecasts from the Chronos model reused as extra inputs to other models.

All engineered features are collected into a structured configuration file so inference code can reproduce them.

---

## 6. The Three Model Families

The notebook uses **three different forecasting approaches** and later blends them. Each has complementary strengths.

### 6.1 Chronos-T5 Time-Series Foundation Model

- Chronos is a **pretrained time-series transformer model** (from Amazon) that has seen many generic time series.
- Internally, it looks similar to models used for natural language:
  - An **encoder** reads a sequence of past values (our context window).
  - A **decoder** learns to produce the next values step by step.
- Attention layers allow the model to "look back" at *all* past positions when predicting the future, not just the most recent few.

In this notebook, we use Chronos in two main phases:

1. **Warm-up phase** (optional but recommended on small datasets):
   - Most of the pretrained weights are **frozen**.
   - Only a small number of parameters (typically the output head and a few higher layers) are updated.
   - This stabilises training and prevents catastrophic forgetting of general time-series knowledge.

2. **Fine-tune phase**:
   - More layers are **unfrozen**.
   - The learning rate is reduced.
   - The model adapts specifically to Dindigul indoor greenhouse patterns (seasonal structure, day–night cycles, etc.).

For each target variable, Chronos sees a **fixed-length context window** (for example, the last 30 days of that variable and related features) and learns to forecast the next two steps (24h, 48h). These Chronos predictions are:

- Used directly as one component in the ensemble.
- Reused as **meta-features** for XGBoost and LSTM (giving them a "head start" from the foundation model).

**Intuition:** Chronos is like a "time-series language model" — it has a general understanding of how sequences behave (trends, seasonality, shocks), and we gently fine-tune it so that its "language" becomes the micro-climate of this specific greenhouse.

### 6.2 XGBoost Gradient-Boosted Trees

- XGBoost is a **gradient-boosted decision tree** method designed for structured, tabular data.
- It builds an **ensemble of small decision trees**, where each new tree focuses on correcting the errors of the previous trees.

Conceptually:

1. Start with a simple model (e.g. predicting the mean of the training targets).
2. Compute the **residuals** (the mistakes the model makes).
3. Train a small decision tree to predict those residuals.
4. Add this tree to the model (with a learning-rate multiplier).
5. Repeat steps 2–4 many times.

Because each tree focuses on "what the previous model got wrong", the ensemble becomes very flexible and can capture complicated relationships between features and targets.

In our training setup:

- We train **one XGBoost model per (target variable, horizon)** pair. For example:
  - `xgb_temp_24h.pkl`
  - `xgb_temp_48h.pkl`
  - `xgb_humidity_24h.pkl`
  - `xgb_humidity_48h.pkl`, etc.
- Each model sees the full **feature vector** for the last available day (lags, rolling statistics, season flags, Chronos meta-features, etc.).

To make training stable on limited data, the notebook typically uses:

- A **warm-up phase** with:
  - Shallow trees
  - Strong regularisation (e.g. larger `min_child_weight`, `gamma`)
  - Fewer boosting rounds
- A **fine-tune phase** with:
  - Slightly deeper trees
  - Relaxed regularisation where safe
  - Early stopping on a validation set (stop when further boosting does not improve validation error).

This gives a set of XGBoost models that are strong on "instantaneous" tabular patterns (e.g. particular lag combinations, seasonal identifiers) and nicely complement the sequence-focused Chronos and LSTM.

### 6.3 LSTM Neural Network

- LSTM stands for **Long Short-Term Memory**, a type of recurrent neural network (RNN) explicitly designed to remember information over many time steps.
- It processes sequences one step at a time while carrying an **internal state** that acts like a memory.

At each time step (e.g. each day), the LSTM receives a **feature vector** containing:

- All engineered features for that day (lags, rolling stats, season flags, etc.).

and updates two internal vectors:

- The **hidden state** (short-term memory).
- The **cell state** (longer-term memory).

Special "gates" (input, forget, output) decide how much of the past to keep and how much of the new information to store.

#### How We Use LSTM in This Project

1. We build a **sliding window** dataset:
  - For each day in the training period (after enough history is available), we take the previous `context_length` days of features as one input example.
  - The target output is a small vector of future values (24h and 48h ahead) for a given variable.

2. The LSTM is shared across targets, but its **output head** is adapted per target. During training, we:
  - Feed the sliding windows through the LSTM.
  - Take the final hidden state and pass it through a small feed-forward "head" to produce the 24h and 48h forecasts.

3. The notebook uses training techniques suitable for small datasets:
  - A robust loss function such as **Huber loss** (less sensitive to occasional large errors) instead of plain squared error.
  - **Gradient clipping** to prevent exploding gradients.
  - A learning-rate schedule (e.g. cosine annealing) to start with a slightly higher learning rate and then gradually reduce it.
  - Early stopping on validation error.

4. After training, per-target LSTM weights and per-target scalers are **bundled** into a single `.pt` file (`environment_forecast_<run_id>.pt`). This bundle contains:
  - The shared LSTM architecture configuration (input size, hidden size, number of layers, dropout).
  - A state dictionary for each target’s LSTM head.
  - The target scalers used to bring predictions back to the original units.

In this project, LSTM replaces a more complex architecture (Temporal Fusion Transformer, TFT) because:

- The dataset is relatively small (hundreds of samples).
- LSTMs have fewer parameters and are more stable when data is limited.
- Training time and resource requirements are modest, which is better suited for a typical research or applied setting.

### 6.4 How We Train All Models Together

Putting it all together, the training process in the notebook typically follows this order:

1. **Data preparation & splitting**
  - Load the full historical indoor climate dataset.
  - Clean and align timestamps.
  - Split into **training**, **validation**, and **test** sets along the time axis (no shuffling, to respect temporal order).

2. **Feature engineering (once)**
  - Compute all lags, rolling statistics, season flags, climate normals, anomalies, and solar/Chronos meta-features.
  - Store the feature configuration and scalers so the same transformations can be applied later during inference.

3. **Chronos training**
  - Build sequences of length `chronos_context_length` from the training portion.
  - Run **warm-up epochs** with most weights frozen (stabilisation).
  - Run **fine-tuning epochs** with more layers unfrozen, monitoring validation loss.
  - Save the best Chronos weights and the fine-tuning configuration.

4. **XGBoost training**
  - For each (target, horizon) pair, construct a tabular dataset from the engineered features (typically, one row per time step with all features and a shifted target column).
  - Train a warm-up model, then a full model with early stopping.
  - Save each trained model to its own `xgb_<target>_<horizon>.pkl` file.

5. **LSTM training**
  - Build sliding windows of feature sequences for each target.
  - Train the shared LSTM plus per-target heads using Huber loss, gradient clipping, and a decaying learning rate.
  - Track validation metrics and keep the best-performing checkpoint.
  - Bundle all trained LSTM weights and scalers into `environment_forecast_<run_id>.pt`.

6. **Ensemble weight search**
  - For each (target, horizon) pair, collect validation predictions from Chronos, XGBoost, and LSTM.
  - Use Optuna to search over combinations of `(w_chronos, w_xgb, w_lstm)` that minimise validation error.
  - Store the best weights in `ensemble_weights.json`.

7. **Final evaluation and export**
  - Run the full ensemble (plus conditions classifiers) on the **test set** only.
  - Compute metrics (MAE, RMSE, R², accuracy for conditions) and store them in `evaluation_metrics.json`.
  - Generate summary plots and move them into the artefacts `plots/` directory.
  - Clean up intermediate checkpoints, leaving only the main realtime bundle and artefacts required for deployment.

This training schedule ensures that:

- All models see **consistent features and splits**.
- Chronos and LSTM exploit temporal structure, while XGBoost focuses on rich tabular interactions.
- The ensemble learns **data-driven weights**, instead of manually guessing how much to trust each component.

---

## 7. Ensemble: Blending the Three Models

No single model is perfect. Instead of choosing just one, the notebook uses an **ensemble**:

- For each **target variable** and **horizon** (24h, 48h), it learns a set of **weights**:
  - Weight for Chronos prediction.
  - Weight for XGBoost prediction.
  - Weight for LSTM prediction.

An optimisation tool (Optuna) searches for weights that minimise error on the validation data.

Final prediction for a given target and horizon is:

```text
final_prediction =
    w_chronos * chronos_forecast
  + w_xgb    * xgboost_forecast
  + w_lstm   * lstm_forecast
```

These weights are saved in `ensemble_weights.json` and used later in the inference loader.

---

## 8. Conditions Classifier (Sky Condition Labels)

Numbers like "26.7 °C" and "65% humidity" are useful, but sometimes we want a **human-friendly label** such as:

- "Clear / Sunny"
- "Partly Cloudy"
- "Overcast"

The notebook trains a **separate classifier** (typically a Random Forest) that:

1. Takes near-future predicted weather values for 24h/48h horizons.
2. Assigns them to discrete condition labels using a `LabelEncoder`.

This classifier is saved as:

- `conditions_classifier_24h.pkl`
- `conditions_classifier_48h.pkl`

and the encoder is stored in `label_encoder.pkl`.

---

## 9. What Gets Saved After Training?

After the notebook finishes, you should have a structure like:

```text
src/agritwin_gh/models/
├── environment_forecast_<run_id>.pt            ← primary LSTM bundle (all targets)
└── artifacts/environment_forecast_<run_id>/
    ├── feature_config.json                      ← all feature names & settings
    ├── scalers.pkl                              ← feature scaler (e.g. RobustScaler)
    ├── label_encoder.pkl                        ← LabelEncoder for conditions
    ├── climate_normals.json                     ← monthly/weekly climate normals
    ├── ensemble_weights.json                    ← ensemble weights per target/horizon
    ├── evaluation_metrics.json                  ← test metrics (MAE, RMSE, R², etc.)
    ├── xgb_<target>_<horizon>.pkl              ← XGBoost models
    ├── conditions_classifier_<horizon>.pkl     ← condition classifiers
    ├── chronos_finetuned/
    │   ├── t5_finetuned_state_dict.pt
    │   └── chronos_finetune_config.json
    ├── environment_forecast_loader.py           ← inference helper module
    └── plots/                                   ← visualisations
```

> **Important:** The notebook also performs cleanup so that only the **final realtime bundle** (`environment_forecast_<run_id>.pt`) and the necessary artefacts stay on disk.

---

## 10. How to Use the Final Model for Inference

The notebook writes a reusable Python helper called `environment_forecast_loader.py`. It contains a class (for example) `EnvironmentForecastModel` that:

- Knows how to load:
  - Feature scaler
  - Label encoder
  - Ensemble weights
  - XGBoost models
  - Condition classifiers
  - Chronos finetuned weights
  - LSTM bundle from the `.pt` file
- Exposes a simple method: `predict(df_context)`.

### 10.1 Minimal Usage Example

Below is an example (adapted from the notebook) of how you would use the loader in your own code.

```python
from src.agritwin_gh.models.artifacts.environment_forecast_<run_id>.environment_forecast_loader import (
    EnvironmentForecastModel,
)

artifacts_dir = "src/agritwin_gh/models/artifacts/environment_forecast_<run_id>"
model_path   = "src/agritwin_gh/models/environment_forecast_<run_id>.pt"

# Create model instance (CPU by default)
model = EnvironmentForecastModel(
    artifacts_dir=artifacts_dir,
    main_model_path=model_path,
    device="cpu",
)

# df_last_30_rows must contain at least `context_length` rows and
# all feature + target columns named in feature_config.json
preds = model.predict(df_last_30_rows)

print(preds)
# Example output:
# {"temp": {"24h": 28.3, "48h": 29.1}, "humidity": {"24h": 67.8, "48h": 65.2}, ...}
```

> **Tip:** `feature_config.json` inside the artefacts directory tells you:
> - `all_feature_names` – all feature columns you must provide.
> - `target_cols` – the variables that are being predicted.
> - `context_length` – how many recent rows are required.

---

## 11. Realtime Test Script

To make it easier to validate the exported model outside the notebook, there is a script:

- `scripts/test_environment_forecast_realtime.py`

### 11.1 What the Test Script Does

- Locates the latest `environment_forecast_<run_id>` artefacts directory (or uses the one you specify).
- Loads `environment_forecast_loader.py` from that directory.
- Instantiates `EnvironmentForecastModel` using the exported `.pt` bundle only (no in-memory notebook state).
- Builds a **test context window**:
  - Either from a CSV file you provide, or
  - From a synthetic test case built inside the script.
- Calls `model.predict(context_df)`.
- Prints 24h/48h predictions for each target.

### 11.2 How to Run It

**A. With a CSV file of recent observations:**

```bash
python scripts/test_environment_forecast_realtime.py \
  --context-csv path/to/latest_weather.csv
```

**B. With explicit artefact and model paths:**

```bash
python scripts/test_environment_forecast_realtime.py \
  --context-csv path/to/latest_weather.csv \
  --artifacts-dir src/agritwin_gh/models/artifacts/environment_forecast_<run_id> \
  --model-path   src/agritwin_gh/models/environment_forecast_<run_id>.pt
```

**C. With a built-in synthetic test case (no CSV needed):**

```bash
python scripts/test_environment_forecast_realtime.py --run-testcase
```

The script prints which source it used for context (CSV vs synthetic) and lists predictions per target.

---

## 12. Understanding the Metrics and Plots

The notebook saves several plots (under the artefacts `plots/` folder) and a metrics file.

### 12.1 Metrics in `evaluation_metrics.json`

For each (target, horizon) pair you will typically see:

- **MAE (Mean Absolute Error)** – average absolute difference between prediction and actual value.
- **RMSE (Root Mean Squared Error)** – penalises larger errors more.
- **R² Score** – how well the model explains variance (1.0 is perfect, 0.0 means "no better than constant").

You may also see accuracy-like measures for condition classifiers.

### 12.2 Key Plots

Common plots include:

- **EDA time series plot** – visual inspection of raw historical data.
- **Seasonal boxplots** – how variables vary by season or month.
- **Correlation heatmap** – relationships between variables.
- **Training curves** for Chronos and LSTM – show how loss decreases over epochs.
- **Ensemble prediction vs actual** on the test set.
- **Metrics summary bar charts** – side-by-side comparison of 24h and 48h performance.

These plots help answer questions like:

- "Does the model systematically under- or over-predict in certain seasons?"
- "Are 48h forecasts much harder than 24h?"
- "Which variables are hardest to predict?"

---

## 13. Typical Realtime Scenario Walkthrough

Let’s imagine how this system is used in production.

1. **Sensors collect data** every day (or hour) for indoor temperature, humidity, etc.
2. At a chosen time (e.g. midnight), the latest readings are appended to a central database.
3. A small job (Python script or API call) does the following:
   - Fetches the last `context_length` rows of data.
   - Assembles them into a `DataFrame` with the required columns.
   - Loads `EnvironmentForecastModel` from artefacts.
   - Calls `predict(df_context)`.
4. The model returns 24h/48h forecasts for each target variable.
5. Optionally, the conditions classifier turns these into labels ("Sunny", "Cloudy", etc.).
6. The **AgriTwin-GH digital twin** consumes these predictions to:
   - Update plant growth and disease risk models.
   - Suggest control actions (fan speed, heater on/off, fogging, etc.).
   - Feed dashboards and alerts for growers.

From the operator’s point of view, they see **tomorrow’s conditions and risks** on a dashboard, without having to understand any of the underlying machine-learning details.

---

## 14. Common Questions (FAQ)

**Q1. Do I need to retrain the model often?**  
Not necessarily. If your greenhouse structure and location stay the same, you can retrain periodically (e.g. every season or every few months) when you have more data. If you change hardware (e.g. sensors, insulation), a fresh training run is recommended.

**Q2. Can this work in a different location than Dindigul?**  
Yes, but you must retrain with local data. Season labels and climate normals are specific to each location. The code is written to be adaptable: you can plug in your own dataset and region-specific season definitions.

**Q3. What happens if some sensor values are missing?**  
The notebook includes basic cleaning and imputation strategies. For realtime deployment, you should ensure that missing values are handled (e.g. via last-observation-carried-forward) before calling `predict`.

**Q4. Why use three model types instead of just one?**  
Each model family has strengths:
- Chronos brings pretrained sequence knowledge.
- XGBoost is strong on tabular patterns and interactions.
- LSTM is good at capturing medium-range temporal structure.

The ensemble usually performs more robustly than any single component.

**Q5. Can I run inference on GPU?**  
Yes. The loader accepts a `device` argument (e.g. `"cuda"` if you have a compatible GPU). For typical greenhouse workloads, CPU is usually sufficient.

---

## 15. Glossary

- **Time Series** – A sequence of measurements taken over time (e.g. daily temperature).
- **Horizon** – How far into the future we are predicting (24h, 48h, etc.).
- **Feature** – An input variable to a model (e.g. day-of-year, lagged temperature).
- **Target Variable** – The thing we want to predict (e.g. tomorrow’s temperature).
- **Chronos** – A pretrained transformer-based model for time-series forecasting.
- **XGBoost** – A popular gradient-boosted decision tree model for tabular data.
- **LSTM** – A type of neural network that processes sequences and remembers context.
- **Ensemble** – A combination of multiple models; predictions are blended, often improving accuracy.
- **Scaler** – A transformation that normalises input data (e.g. centering and scaling).
- **Label Encoder** – A mechanism to convert string labels ("Sunny", "Cloudy") into numeric codes.
- **MAE / RMSE / R²** – Standard error and performance metrics for regression tasks.

---

This document is designed to be read alongside the `weather_forecast.ipynb` notebook.  
You can treat the notebook as the *implementation* and this markdown file as the *guided tour* and reference manual for the greenhouse weather forecasting model.
