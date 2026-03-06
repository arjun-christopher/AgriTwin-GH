# Evaluation Summary — `growth_progression_20260306_212928`

## Classifier — Next Growth Stage

| Split | Model | Accuracy | F1 macro | F1 weighted | Precision macro | Recall macro |
|---|---|---|---|---|---|---|
| validation | RF | 0.071 | 0.0591 | 0.1209 | 0.1981 | 0.0347 |
| validation | LSTM | 0.0061 | 0.0067 | 0.0115 | 0.0514 | 0.0036 |
| test | RF | 0.3607 | 0.2695 | 0.3266 | 0.2542 | 0.2936 |
| test | LSTM | 0.1452 | 0.0882 | 0.1162 | 0.0824 | 0.1066 |

## Regressor — Time to Next Transition (hours)

| Split | Model | MAE (h) | RMSE (h) | R² | MAPE (%) |
|---|---|---|---|---|---|
| validation | RF | 131.5581 | 155.1777 | 0.1084 | 316.8054 |
| validation | LSTM | 134.153 | 164.0721 | 0.0033 | 278.1106 |
| test | RF | 126.2463 | 155.5257 | 0.2234 | 296.6995 |
| test | LSTM | 150.8724 | 178.36 | -0.0214 | 277.7906 |

## Selection

**Selected model:** `RF`

**Reason:** RF strictly better on both clf (>5% margin) and reg

**Composite score (0.6 clf + 0.4 reg):** -0.5401
