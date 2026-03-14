
"""
Inference loader for environment_forecast_20260313_210748
Usage:
    from environment_forecast_loader import EnvironmentForecastModel
    model = EnvironmentForecastModel("<artifacts_dir>", main_model_path="<path>.pt")
    preds = model.predict(df_last_30_days)    # Returns dict: col -> {"24h": val, "48h": val}
"""
import joblib, json, torch, torch.nn as nn, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from chronos import ChronosPipeline


# ─ Inline LSTM definition (must match training architecture) ──────────────────
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, n_layers=2,
                 dropout=0.3, pred_len=2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, n_layers,
                               batch_first=True,
                               dropout=(dropout if n_layers > 1 else 0.0))
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(self.norm(out[:, -1])))


class EnvironmentForecastModel:
    def __init__(self, artifacts_dir, main_model_path=None, device="cpu"):
        self.art    = Path(artifacts_dir)
        self.device = device

        # ── Scalers / encoders ────────────────────────────────────────────────
        self.feat_scaler   = joblib.load(self.art / "scalers.pkl")
        self.label_encoder = joblib.load(self.art / "label_encoder.pkl")

        # ── Config ────────────────────────────────────────────────────────────
        with open(self.art / "feature_config.json") as f:
            self.feat_cfg = json.load(f)
        with open(self.art / "ensemble_weights.json") as f:
            self.ens_weights = json.load(f)

        # ── XGBoost models ────────────────────────────────────────────────────
        self.xgb_models = {}
        for col in self.feat_cfg["target_cols"]:
            for h in ["24h", "48h"]:
                p = self.art / f"xgb_{col}_{h}.pkl"
                if p.exists():
                    self.xgb_models[(col, h)] = joblib.load(p)

        # ── Conditions classifiers ─────────────────────────────────────────────
        self.cond_models = {}
        for h in ["24h", "48h"]:
            p = self.art / f"conditions_classifier_{h}.pkl"
            if p.exists():
                self.cond_models[h] = joblib.load(p)

        # ── LSTM bundle ───────────────────────────────────────────────────────
        self.lstm_models         = {}
        self.lstm_target_scalers = {}
        if main_model_path and Path(main_model_path).exists():
            bundle = torch.load(main_model_path, map_location=device)
            cfg    = bundle["lstm_config"]
            for col, state in bundle["lstm_states"].items():
                m = WeatherLSTM(
                    cfg["input_size"], cfg["hidden_size"],
                    cfg["n_layers"],   cfg["dropout"], cfg["pred_len"]
                ).to(device)
                m.load_state_dict(state)
                m.eval()
                self.lstm_models[col] = m
            for col, sc in bundle["target_scalers"].items():
                self.lstm_target_scalers[col] = sc

        # ── Chronos ───────────────────────────────────────────────────────────
        self.chronos = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small", device_map=device, dtype=torch.float32)
        chron_sd = self.art / "chronos_finetuned" / "t5_finetuned_state_dict.pt"
        if chron_sd.exists():
            self.chronos.model.model.load_state_dict(
                torch.load(chron_sd, map_location=device))
        self.chronos.model.eval()

    def _chronos_predict(self, series_np):
        """Return [2] array: [24h, 48h] Chronos median forecasts."""
        ctx = torch.tensor(series_np[-30:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            fc = self.chronos.predict(ctx, prediction_length=2, num_samples=50)
        return torch.quantile(fc.squeeze(0), 0.5, dim=0).cpu().numpy()

    def predict(self, df_context):
        """
        df_context: DataFrame with at least context_length rows and all
                    FEATURE_COLS + TARGET_COLS present.
        Returns dict: col -> {"24h": float, "48h": float}
        """
        feat_cols  = self.feat_cfg["all_feature_names"]
        target_cols = self.feat_cfg["target_cols"]
        ctx_len     = self.feat_cfg["context_length"]
        result      = {}

        # Feature matrix for last ctx_len rows (scaled)
        Xraw = df_context[feat_cols].fillna(0).values[-ctx_len:]
        Xsc  = self.feat_scaler.transform(Xraw)         # [ctx_len, F]
        Xt   = torch.tensor(Xsc[np.newaxis], dtype=torch.float32).to(self.device)

        for col in target_cols:
            w_key_24 = f"{col}_24h"
            w_key_48 = f"{col}_48h"
            w24 = self.ens_weights.get(w_key_24, {"chronos": 0.33, "xgb": 0.33, "lstm": 0.34})
            w48 = self.ens_weights.get(w_key_48, {"chronos": 0.33, "xgb": 0.33, "lstm": 0.34})

            # Chronos
            chron = self._chronos_predict(df_context[col].values)

            # XGBoost
            xgb_24 = float(self.xgb_models[(col, "24h")].predict(Xsc[-1:])[0])                      if (col, "24h") in self.xgb_models else chron[0]
            xgb_48 = float(self.xgb_models[(col, "48h")].predict(Xsc[-1:])[0])                      if (col, "48h") in self.xgb_models else chron[1]

            # LSTM
            if col in self.lstm_models:
                with torch.no_grad():
                    lstm_raw = self.lstm_models[col](Xt).cpu().numpy()[0]  # [2]
                sc = self.lstm_target_scalers.get(col)
                if sc is not None:
                    lstm_preds = sc.inverse_transform(
                        lstm_raw.reshape(-1, 1)).ravel()
                else:
                    lstm_preds = lstm_raw
                lstm_24, lstm_48 = float(lstm_preds[0]), float(lstm_preds[1])
            else:
                lstm_24, lstm_48 = chron[0], chron[1]

            pred_24 = (w24["chronos"] * float(chron[0])
                     + w24["xgb"]     * xgb_24
                     + w24["lstm"]    * lstm_24)
            pred_48 = (w48["chronos"] * float(chron[1])
                     + w48["xgb"]     * xgb_48
                     + w48["lstm"]    * lstm_48)

            result[col] = {"24h": round(pred_24, 4), "48h": round(pred_48, 4)}

        return result
