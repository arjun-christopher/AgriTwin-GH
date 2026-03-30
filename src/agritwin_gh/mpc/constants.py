"""
Canonical labels, index maps, and physics constants for the MPC control layer.

This is the single source of truth — no other MPC file hard-codes stage or
disease strings.
"""

from __future__ import annotations

# ── Growth stages (ordered, canonical) ────────────────────────────────────────

GROWTH_STAGES: tuple[str, ...] = (
    "seedling",
    "early vegetative",
    "flowering initiation",
    "flowering",
    "unripe",
    "ripe",
)

GROWTH_STAGE_INDEX: dict[str, int] = {s: i for i, s in enumerate(GROWTH_STAGES)}

# DB `stage_name` uses underscores; canonical labels use spaces.
GROWTH_STAGE_TO_DB: dict[str, str] = {
    "seedling": "seedling",
    "early vegetative": "early_vegetative",
    "flowering initiation": "flowering_initiation",
    "flowering": "flowering",
    "unripe": "unripe",
    "ripe": "ripe",
}
GROWTH_STAGE_FROM_DB: dict[str, str] = {v: k for k, v in GROWTH_STAGE_TO_DB.items()}

# image_metadata.subcategory values used for random image retrieval.
GROWTH_STAGE_IMAGE_SUBCATEGORY: dict[str, str] = {
    "seedling": "stage1_seedling",
    "early vegetative": "stage2_early_vegetative",
    "flowering initiation": "stage3_flowering_initiation",
    "flowering": "stage4_flowering",
    "unripe": "stage5_unripe",
    "ripe": "stage6_ripe",
}

# ── Disease categories (canonical) ───────────────────────────────────────────

DISEASE_CATEGORIES: tuple[str, ...] = (
    "powdery mildew",
    "spider mites",
    "leaf mold",
    "early blight",
    "late blight",
    "healthy leaves",
)

DISEASE_CATEGORY_INDEX: dict[str, int] = {d: i for i, d in enumerate(DISEASE_CATEGORIES)}

# DB `disease_name` column values (underscore form).
DISEASE_TO_DB: dict[str, str] = {
    "powdery mildew": "powdery_mildew",
    "spider mites": "spider_mites",
    "leaf mold": "leaf_mold",
    "early blight": "early_blight",
    "late blight": "late_blight",
    "healthy leaves": "healthy_leaves",
}
DISEASE_FROM_DB: dict[str, str] = {v: k for k, v in DISEASE_TO_DB.items()}

# image_metadata.subcategory values for random image retrieval.
DISEASE_IMAGE_SUBCATEGORY: dict[str, str] = {
    "powdery mildew": "tomato_powdery_mildew",
    "spider mites": "tomato_spider_mites",
    "leaf mold": "tomato_leaf_mold",
    "early blight": "tomato_early_blight",
    "late blight": "tomato_late_blight",
    "healthy leaves": "tomato_leaf_healthy",
}

# Combined image subcategory map (growth + disease) for convenience.
IMAGE_SUBCATEGORY_MAP: dict[str, dict[str, str]] = {
    "growth_stage": GROWTH_STAGE_IMAGE_SUBCATEGORY,
    "disease": DISEASE_IMAGE_SUBCATEGORY,
}

# ── MPC state & control variable names (ordered) ─────────────────────────────

STATE_VARIABLES: tuple[str, ...] = (
    "indoor_temp",
    "indoor_humidity",
    "soil_moisture",
    "co2",
    "light_intensity",
    "disease_risk_score",
    "growth_stage_index",
    "vpd",
    "leaf_wetness_proxy",
)

CONTROL_VARIABLES: tuple[str, ...] = (
    "fan_speed",
    "vent_opening",
    "irrigation_qty",
    "heater_output",
    "led_intensity",
    "co2_valve_pct",
    "fogger_duty",
)

# ── Timing constants ─────────────────────────────────────────────────────────

DT_MINUTES: int = 5
STEPS_PER_HOUR: int = 60 // DT_MINUTES  # 12

# ── Alert levels ──────────────────────────────────────────────────────────────

ALERT_GREEN = "GREEN"
ALERT_YELLOW = "YELLOW"
ALERT_RED = "RED"

# ── Helpers ───────────────────────────────────────────────────────────────────


def stage_label_to_index(label: str) -> int:
    """Convert a canonical growth-stage label to its integer index (0-5)."""
    try:
        return GROWTH_STAGE_INDEX[label]
    except KeyError:
        raise ValueError(
            f"Unknown growth stage '{label}'. "
            f"Valid stages: {GROWTH_STAGES}"
        )


def stage_index_to_label(index: int) -> str:
    """Convert an integer index (0-5) to its canonical growth-stage label."""
    if 0 <= index < len(GROWTH_STAGES):
        return GROWTH_STAGES[index]
    raise ValueError(
        f"Stage index {index} out of range [0, {len(GROWTH_STAGES) - 1}]."
    )


def disease_label_to_db(label: str) -> str:
    """Convert a canonical disease label to its DB column-compatible name."""
    try:
        return DISEASE_TO_DB[label]
    except KeyError:
        raise ValueError(
            f"Unknown disease '{label}'. "
            f"Valid diseases: {DISEASE_CATEGORIES}"
        )


def disease_db_to_label(db_name: str) -> str:
    """Convert a DB disease_name value to its canonical label."""
    try:
        return DISEASE_FROM_DB[db_name]
    except KeyError:
        raise ValueError(
            f"Unknown DB disease name '{db_name}'. "
            f"Known DB names: {list(DISEASE_FROM_DB.keys())}"
        )


def growth_stage_db_to_label(db_name: str) -> str:
    """Convert a DB stage_name value to its canonical label."""
    try:
        return GROWTH_STAGE_FROM_DB[db_name]
    except KeyError:
        raise ValueError(
            f"Unknown DB stage name '{db_name}'. "
            f"Known DB names: {list(GROWTH_STAGE_FROM_DB.keys())}"
        )


def sigmoid(value: float, threshold: float, steepness: float = 0.2) -> float:
    """Logistic sigmoid centred on *threshold* — returns risk ∈ [0, 1]."""
    import math
    return 1.0 / (1.0 + math.exp(-steepness * (value - threshold)))


def compute_vpd(temp_c: float, rh_pct: float) -> float:
    """Compute Vapor Pressure Deficit (kPa) via the Tetens equation."""
    svp = 0.6108 * _exp_tetens(temp_c)
    return svp * (1.0 - rh_pct / 100.0)


def compute_dew_point(temp_c: float, rh_pct: float) -> float:
    """Approximate dew-point temperature (°C) using the Magnus formula."""
    import math
    a, b = 17.27, 237.7
    alpha = (a * temp_c) / (b + temp_c) + math.log(max(rh_pct, 1.0) / 100.0)
    return (b * alpha) / (a - alpha)


def compute_leaf_wetness_proxy(humidity: float, temp_c: float,
                                dew_point: float) -> float:
    """Simple proxy for leaf wetness (0-1) based on humidity and dew-point gap."""
    gap = temp_c - dew_point
    hum_factor = max(0.0, min(1.0, (humidity - 60.0) / 35.0))
    gap_factor = max(0.0, min(1.0, 1.0 - gap / 10.0))
    return round(0.6 * hum_factor + 0.4 * gap_factor, 4)


def compute_disease_risk_score(
    temp: float,
    humidity: float,
    leaf_wetness: float,
    growth_stage: str | None = None,
) -> float:
    """Unified disease risk score (0-1) using sigmoid sub-scores.

    Matches the rule-based approach from feature-demo Notebook 03.
    """
    hum_risk = sigmoid(humidity, 75.0, 0.2)
    wet_risk = sigmoid(leaf_wetness, 0.5, 8.0)
    temp_risk = sigmoid(temp, 22.0, 0.15)

    base = 0.4 * hum_risk + 0.35 * wet_risk + 0.25 * temp_risk

    # Stage multiplier — flowering / unripe / ripe are more vulnerable.
    multiplier = 1.0
    if growth_stage in ("flowering initiation", "flowering", "unripe"):
        multiplier = 1.2
    elif growth_stage == "ripe":
        multiplier = 1.1

    return round(min(1.0, base * multiplier), 4)


# ── Enhanced disease risk scoring ─────────────────────────────────────────────

# Per-stage disease sensitivity multipliers (aligned with StageControlProfile).
_STAGE_DISEASE_SENSITIVITY: dict[str, float] = {
    "seedling": 1.0,
    "early vegetative": 1.0,
    "flowering initiation": 1.3,
    "flowering": 1.5,
    "unripe": 1.4,
    "ripe": 0.8,
}


def compute_enhanced_disease_risk(
    *,
    temp: float,
    humidity: float,
    leaf_wetness: float,
    vpd: float | None = None,
    growth_stage: str | None = None,
    classification_prob: float = 0.0,
    current_severity: float = 0.0,
    severity_24h: float = 0.0,
    severity_48h: float = 0.0,
) -> dict[str, float]:
    """Comprehensive disease risk score combining environmental signals,
    classification outputs, and severity forecasts.

    Returns a dict with named sub-scores **and** the composite ``risk``
    value (0-1) so the dashboard can explain each contribution.

    Sub-scores
    ----------
    env_humidity : sigmoid on RH (weight 0.20)
    env_leaf_wet : sigmoid on leaf wetness (weight 0.15)
    env_temp     : sigmoid on temperature (weight 0.10)
    env_vpd      : inverse sigmoid on VPD — low VPD ⇒ high risk (weight 0.10)
    classification : disease-class probability from the image classifier (weight 0.15)
    severity_current : current average severity across diseases (weight 0.10)
    severity_trend : max(sev_24h, sev_48h) normalised to 0-1 (weight 0.10)
    stage_sensitivity : multiplicative, not additive (applied to subtotal)

    Composite: ``risk = clip(subtotal * stage_sensitivity, 0, 1)``
    """
    # ── Environmental sub-scores ──────────────────────────────────────
    env_humidity = sigmoid(humidity, 75.0, 0.2)
    env_leaf_wet = sigmoid(leaf_wetness, 0.5, 8.0)
    env_temp = sigmoid(temp, 22.0, 0.15)

    if vpd is not None:
        # Low VPD (< 0.5 kPa) means stagnant, humid air → high risk
        env_vpd = 1.0 - sigmoid(vpd, 0.8, 5.0)
    else:
        env_vpd = 0.0

    # ── Classification / severity sub-scores ──────────────────────────
    classification_score = max(0.0, min(1.0, classification_prob))
    severity_cur = max(0.0, min(1.0, current_severity / 100.0))
    severity_trend = max(0.0, min(1.0, max(severity_24h, severity_48h) / 100.0))

    # ── Weighted sum ──────────────────────────────────────────────────
    subtotal = (
        0.20 * env_humidity
        + 0.15 * env_leaf_wet
        + 0.10 * env_temp
        + 0.10 * env_vpd
        + 0.15 * classification_score
        + 0.10 * severity_cur
        + 0.10 * severity_trend
        # remaining 0.10 headroom absorbed by stage_sensitivity
    )

    # ── Stage sensitivity ─────────────────────────────────────────────
    stage_mult = _STAGE_DISEASE_SENSITIVITY.get(growth_stage or "", 1.0)
    risk = max(0.0, min(1.0, subtotal * stage_mult))

    return {
        "env_humidity": round(env_humidity, 4),
        "env_leaf_wet": round(env_leaf_wet, 4),
        "env_temp": round(env_temp, 4),
        "env_vpd": round(env_vpd, 4),
        "classification": round(classification_score, 4),
        "severity_current": round(severity_cur, 4),
        "severity_trend": round(severity_trend, 4),
        "stage_sensitivity": round(stage_mult, 2),
        "risk": round(risk, 4),
    }


# ── Private ───────────────────────────────────────────────────────────────────

def _exp_tetens(temp_c: float) -> float:
    """Tetens exponential component for SVP calculation."""
    import math
    return math.exp((17.27 * temp_c) / (temp_c + 237.3))
