"""
MPC Input Preparation — database query logic.

**This file is deliberately separated** because it depends on the DB schema
which may evolve.  All other MPC files consume its output through clean
dict / DataFrame interfaces and never import ORM models directly.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from ..models.timeseries import (
    DiseaseProgressionData,
    GreenhouseData,
    GrowthProgressionHourly,
    WeatherData,
)
from .constants import IMAGE_SUBCATEGORY_MAP

logger = logging.getLogger(__name__)


class MPCInputPreparation:
    """Encapsulates every DB query the MPC control layer needs.

    Parameters
    ----------
    session:
        A live SQLAlchemy session (obtained via ``get_db_session()``).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Greenhouse snapshot ────────────────────────────────────────────

    def get_latest_greenhouse_row(self) -> dict[str, Any] | None:
        """Return the most recent ``greenhouse_data`` row as a dict.

        Returns ``None`` if the table is empty.
        """
        row = (
            self._session.query(GreenhouseData)
            .order_by(GreenhouseData.datetime.desc())
            .first()
        )
        if row is None:
            return None
        return {
            "datetime": row.datetime,
            "indoor_temp": row.indoor_temp,
            "indoor_humidity": row.indoor_humidity,
            "indoor_air_velocity": row.indoor_air_velocity,
            "indoor_co2": row.indoor_co2,
            "solarradiation": row.solarradiation,
            "day_night_flag": row.day_night_flag,
            "vpd": row.vpd,
            "dew_point": row.dew_point,
            "leaf_wetness_proxy": row.leaf_wetness_proxy,
        }

    # ── Weather context ────────────────────────────────────────────────

    def get_weather_context(self, lookback_days: int = 30) -> pd.DataFrame:
        """Return the last *lookback_days* of ``weather_data`` as a DataFrame.

        The result is sorted by ``datetime`` ascending and ready to be passed
        to ``EnvironmentForecastModel.predict(df_context)``.
        """
        rows = (
            self._session.query(WeatherData)
            .order_by(WeatherData.datetime.desc())
            .limit(lookback_days * 24)
            .all()
        )
        if not rows:
            return pd.DataFrame()

        records = [
            {
                "datetime": r.datetime,
                "temp": r.temp,
                "humidity": r.humidity,
                "windspeed": r.windspeed,
                "solarradiation": r.solarradiation,
                "conditions": r.conditions,
            }
            for r in rows
        ]
        df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
        return df

    # ── Growth progression context ─────────────────────────────────────

    def get_growth_progression_context(
        self, sequence_length: int = 48
    ) -> pd.DataFrame:
        """Return the last *sequence_length* rows from ``growth_progression_hourly``
        for the latest ``cycle_id``, suitable as input to the growth-stage
        progression LSTM.
        """
        # Find newest cycle_id
        latest_cycle = (
            self._session.query(func.max(GrowthProgressionHourly.cycle_id))
            .scalar()
        )
        if latest_cycle is None:
            return pd.DataFrame()

        rows = (
            self._session.query(GrowthProgressionHourly)
            .filter(GrowthProgressionHourly.cycle_id == latest_cycle)
            .order_by(GrowthProgressionHourly.timestamp.desc())
            .limit(sequence_length)
            .all()
        )
        if not rows:
            return pd.DataFrame()

        records = [
            {
                # Metadata (not used as features)
                "timestamp": r.timestamp,
                "cycle_id": r.cycle_id,
                "stage_name": r.stage_name,
                # Calendar features (stored in DB)
                "year": r.year,
                "month": r.month,
                "day_of_year": r.day_of_year,
                "week_of_year": r.week_of_year,
                "hour": r.hour,
                # Stage / progression features
                "stage_index": r.stage_index,
                "hours_in_current_stage": r.hours_in_current_stage,
                "days_in_current_stage": r.days_in_current_stage,
                "stage_duration_hours": r.stage_duration_hours,
                "stage_duration_days": r.stage_duration_days,
                "stage_progress_pct": r.stage_progress_pct,
                "total_cycle_progress_pct": r.total_cycle_progress_pct,
                "estimated_days_to_next_stage": r.estimated_days_to_next_stage,
                "estimated_hours_to_next_stage": r.estimated_hours_to_next_stage,
                "is_stage_transition": r.is_stage_transition,
                "days_from_cycle_start": r.days_from_cycle_start,
                # Environment features
                "indoor_temp": r.indoor_temp,
                "indoor_humidity": r.indoor_humidity,
                "indoor_air_velocity": r.indoor_air_velocity,
                "indoor_co2": r.indoor_co2,
                "solarradiation": r.solarradiation,
                "day_night_flag": r.day_night_flag,
                "vpd": r.vpd,
                "dew_point": r.dew_point,
                "leaf_wetness_proxy": r.leaf_wetness_proxy,
                # Pre-engineered features stored in DB
                "temperature_rolling_mean_24h": r.temperature_rolling_mean_24h,
                "humidity_rolling_mean_24h": r.humidity_rolling_mean_24h,
                "vpd_proxy": r.vpd_proxy,
                "light_period_flag": r.light_period_flag,
                "cumulative_gdd_like_index": r.cumulative_gdd_like_index,
            }
            for r in rows
        ]
        df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        return df

    # ── Disease progression context ────────────────────────────────────

    def get_disease_progression_context(
        self, sequence_length: int = 48
    ) -> pd.DataFrame:
        """Return the last *sequence_length* rows from ``disease_progression``
        for the latest ``cycle_id``.

        The result is in long format (one row per disease per timestamp).
        Pivoting / reshaping is left to the caller so this layer stays
        schema-agnostic about downstream model expectations.
        """
        latest_cycle = (
            self._session.query(func.max(DiseaseProgressionData.cycle_id))
            .scalar()
        )
        if latest_cycle is None:
            return pd.DataFrame()

        rows = (
            self._session.query(DiseaseProgressionData)
            .filter(DiseaseProgressionData.cycle_id == latest_cycle)
            .order_by(DiseaseProgressionData.timestamp.desc())
            .limit(sequence_length)
            .all()
        )
        if not rows:
            return pd.DataFrame()

        records = [
            {
                # ── Identity / time ───────────────────────────────────
                "timestamp": r.timestamp,
                "cycle_id": r.cycle_id,
                "cycle_label": r.cycle_label,
                "season_label": r.season_label,
                "stage_name": r.stage_name,
                "stage_index": r.stage_index,
                "days_from_cycle_start": r.days_from_cycle_start,
                "day_of_year": r.day_of_year,
                "week_of_year": r.week_of_year,
                "hour": r.hour,
                "hours_in_current_stage": r.hours_in_current_stage,
                "stage_progress_pct": r.stage_progress_pct,
                "total_cycle_progress_pct": r.total_cycle_progress_pct,
                "is_stage_transition": r.is_stage_transition,
                # ── Environment ───────────────────────────────────────
                "indoor_temp": r.indoor_temp,
                "indoor_humidity": r.indoor_humidity,
                "indoor_air_velocity": r.indoor_air_velocity,
                "indoor_co2": r.indoor_co2,
                "solarradiation": r.solarradiation,
                "day_night_flag": r.day_night_flag,
                "vpd": r.vpd,
                "dew_point": r.dew_point,
                "leaf_wetness_proxy": r.leaf_wetness_proxy,
                "temperature_rolling_mean_24h": r.temperature_rolling_mean_24h,
                "humidity_rolling_mean_24h": r.humidity_rolling_mean_24h,
                "vpd_proxy": r.vpd_proxy,
                "cumulative_gdd_like_index": r.cumulative_gdd_like_index,
                # ── Disease-specific ──────────────────────────────────
                "disease_name": r.disease_name,
                "disease_present_flag": r.disease_present_flag,
                "disease_risk_score": r.disease_risk_score,
                "current_infection_pct": r.current_infection_pct,
                "infection_growth_rate_hourly": r.infection_growth_rate_hourly,
                "stage_susceptibility_score": r.stage_susceptibility_score,
                "outbreak_trigger_flag": r.outbreak_trigger_flag,
                "control_action_flag": r.control_action_flag,
                "control_action_type": r.control_action_type,
            }
            for r in rows
        ]
        df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        return df

    # ── Current disease severity ───────────────────────────────────────

    def get_current_disease_severity(self) -> dict[str, float]:
        """Return latest ``current_infection_pct`` keyed by ``disease_name``.

        Only rows from the most recent ``cycle_id`` are considered.
        """
        latest_cycle = (
            self._session.query(func.max(DiseaseProgressionData.cycle_id))
            .scalar()
        )
        if latest_cycle is None:
            return {}

        # Sub-query: the most recent timestamp per disease in the latest cycle
        subq = (
            self._session.query(
                DiseaseProgressionData.disease_name,
                func.max(DiseaseProgressionData.timestamp).label("max_ts"),
            )
            .filter(DiseaseProgressionData.cycle_id == latest_cycle)
            .group_by(DiseaseProgressionData.disease_name)
            .subquery()
        )

        rows = (
            self._session.query(
                DiseaseProgressionData.disease_name,
                DiseaseProgressionData.current_infection_pct,
            )
            .join(
                subq,
                (DiseaseProgressionData.disease_name == subq.c.disease_name)
                & (DiseaseProgressionData.timestamp == subq.c.max_ts),
            )
            .filter(DiseaseProgressionData.cycle_id == latest_cycle)
            .all()
        )

        return {name: float(pct or 0.0) for name, pct in rows}

    # ── Image metadata (placeholder interface) ─────────────────────────

    def get_random_image_for_category(
        self, category: str, subcategory: str
    ) -> dict[str, Any] | None:
        """Return a random row from ``image_metadata`` matching
        *category* / *subcategory*.

        This is a **placeholder** — the ``image_metadata`` table schema has not
        been provided.  Implementors should replace the body once the ORM model
        is available.

        Expected return dict::

            {
                "image_key": str,
                "bucket_name": str,
                "file_name": str,
            }
        """
        logger.warning(
            "get_random_image_for_category is a placeholder; "
            "image_metadata ORM model not yet wired."
        )
        return None
