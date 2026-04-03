"""
``agritwin_gh.mpc`` — Model Predictive Control package for AgriTwin-GH.

Public re-exports for convenience.
"""

# ── Constants ─────────────────────────────────────────────────────────────────
from .constants import (
    ALERT_GREEN,
    ALERT_RED,
    ALERT_YELLOW,
    CONTROL_VARIABLES,
    DISEASE_CATEGORIES,
    DISEASE_CATEGORY_INDEX,
    DISEASE_FROM_DB,
    DISEASE_IMAGE_SUBCATEGORY,
    DISEASE_TO_DB,
    DT_MINUTES,
    GROWTH_STAGE_FROM_DB,
    GROWTH_STAGE_IMAGE_SUBCATEGORY,
    GROWTH_STAGE_INDEX,
    GROWTH_STAGE_TO_DB,
    GROWTH_STAGES,
    IMAGE_SUBCATEGORY_MAP,
    STATE_VARIABLES,
    STEPS_PER_HOUR,
    compute_disease_risk_score,
    compute_dew_point,
    compute_enhanced_disease_risk,
    compute_leaf_wetness_proxy,
    compute_vpd,
    disease_db_to_label,
    disease_label_to_db,
    growth_stage_db_to_label,
    stage_index_to_label,
    stage_label_to_index,
)

# ── State dataclasses ─────────────────────────────────────────────────────────
from .state import (
    ActuatorState,
    ComparisonMetrics,
    ControllerDecisionContext,
    DiseaseClassificationOutput,
    DiseaseProgressionOutput,
    DigitalTwinStepPayload,
    DigitalTwinTrajectoryPayload,
    EvaluationSummary,
    FusedState,
    GreenhouseState,
    GrowthClassificationOutput,
    GrowthProgressionOutput,
    ImagePayload,
    MPCState,
    WeatherState,
)

# ── Setpoints ─────────────────────────────────────────────────────────────────
from .setpoints import (
    StageControlProfile,
    StageSetpoint,
    get_all_control_profiles,
    get_all_setpoints,
    get_control_profile,
    get_setpoint,
)

# ── Constraints ───────────────────────────────────────────────────────────────
from .constraints import (
    ConstraintSet,
    get_default_constraints,
    merge_constraints,
    tighten_constraints_for_disease,
)

# ── Configuration ─────────────────────────────────────────────────────────────
from .config import MPCConfig, load_mpc_config

# ── DB query layer ────────────────────────────────────────────────────────────
from .mpc_input_preparation import MPCInputPreparation

# ── Image streaming ───────────────────────────────────────────────────────────
from .image_streamer import ImageStreamer

# ── Shared utilities ──────────────────────────────────────────────────────────
from .utils import discover_latest_artifact

# ── Model orchestration layer ─────────────────────────────────────────────────
from .disturbance import WeatherDisturbanceForecast
from .disease_penalty import DiseaseRiskPenalty
from .growth_weights import GrowthStageWeights
from .state_fusion import StateFusion
from .digital_twin_output import DigitalTwinOutput
from .runner import MPCRunner

# ── Greenhouse transition model ───────────────────────────────────────────────
from .greenhouse_model import GreenhouseModelParams, GreenhouseTransitionModel

# ── Baseline controller ───────────────────────────────────────────────────────
from .baseline_controller import BaselineControlPayload, RuleBasedController

# ── Cost function ─────────────────────────────────────────────────────────────
from .cost_function import CostBuilder, DiseaseContext, StageCost, TerminalCost

# ── Weather adaptation ──────────────────────────────────────────────────
from .weather_adaptation import WeatherAdaptiveModifiers, compute_weather_adaptation

# ── MPC solver ────────────────────────────────────────────────────────────────
from .mpc_solver import MPCSolution, MPCSolver

# ── Explanation ───────────────────────────────────────────────────────────────
from .explanation import ControllerExplanation, ExplanationBuilder, ExplanationEntry

# ── Evaluation metrics ────────────────────────────────────────────────────────
from .evaluation_metrics import (
    ControllerMetricsBundle,
    ControlQualityMetrics,
    DiseaseBurdenMetrics,
    ResourceMetrics,
    SafetyMetrics,
    TrackingMetrics,
    compute_all_metrics,
)

# ── Digital Twin state & interface ─────────────────────────────────────────────
from .dt_state import DTDiagnostics, DTSnapshot, DTStepInput, DTStepOutput
from .dt_engine import DigitalTwinEngine
from .dt_interface import DigitalTwinPlant

# ── DT runtime loop ──────────────────────────────────────────────────────────
from .dt_runtime_prep import (
    build_fused_state,
    build_mpc_solver,
    prepare_growth_stages,
    prepare_initial_state,
    prepare_weather_sequence,
)
from .dt_loop import (
    DTLoop,
    DTLoopStepResult,
    IMAGE_CADENCE_STEPS,
    MPC_CADENCE_STEPS,
    should_force_mpc_update,
)
from .dt_logger import DTLoopLogger, DTLoopRunSummary

# ── DT input / output / image abstractions ────────────────────────────────────
from .dt_input_provider import (
    DTInputProvider,
    ImageObservation,
    SyntheticInputProvider,
    DatabaseInputProvider,
)
from .dt_image_observer import (
    ImageObserver,
    MinIOImageObserver,
    SyntheticImageObserver,
)
from .dt_output_writer import (
    DTOutputWriter,
    JsonFileOutputWriter,
    fanout_step_to_writer,
)

# ── DT artifact manager ──────────────────────────────────────────────────────
from .dt_artifact_manager import DTArtifactManager

# ── Realtime closed-loop core ─────────────────────────────────────────────────
from .realtime_core import (
    RealtimeLoop,
    RealtimeLoopConfig,
    RealtimeMPCInputPreparation,
    RealtimeRunSummary,
    RealtimeStepResult,
    RunRegistry,
    STAGE_DURATION_HOURS,
    PRIOR_STAGE_HOURS,
    ensure_stream_table,
    seed_initial_state,
    write_step_to_stream,
    estimate_energy,
    diurnal_temp,
    diurnal_humidity,
    diurnal_solar,
    default_soil_moisture,
)

# ── Yield proxy ───────────────────────────────────────────────────────────────
from .yield_proxy import (
    YieldProxyWeights,
    YieldProxyResult,
    compute_yield_proxy,
)

# ── Experiment runner ─────────────────────────────────────────────────────────
from .experiment_runner import (
    ComparisonReport,
    ExperimentConfig,
    ExperimentRunner,
    generate_default_growth_stages,
    generate_default_weather,
    make_baseline_adapter,
    make_default_initial_state,
    make_mpc_adapter,
)

# ── Evaluation API & artifacts ────────────────────────────────────────────────
from .evaluation import (
    load_evaluation_report,
    run_evaluation,
    save_evaluation_artifacts,
)

__all__ = [
    # constants
    "GROWTH_STAGES",
    "GROWTH_STAGE_INDEX",
    "GROWTH_STAGE_TO_DB",
    "GROWTH_STAGE_FROM_DB",
    "GROWTH_STAGE_IMAGE_SUBCATEGORY",
    "DISEASE_CATEGORIES",
    "DISEASE_CATEGORY_INDEX",
    "DISEASE_TO_DB",
    "DISEASE_FROM_DB",
    "DISEASE_IMAGE_SUBCATEGORY",
    "IMAGE_SUBCATEGORY_MAP",
    "STATE_VARIABLES",
    "CONTROL_VARIABLES",
    "DT_MINUTES",
    "STEPS_PER_HOUR",
    "ALERT_GREEN",
    "ALERT_YELLOW",
    "ALERT_RED",
    "stage_label_to_index",
    "stage_index_to_label",
    "disease_label_to_db",
    "disease_db_to_label",
    "growth_stage_db_to_label",
    "compute_vpd",
    "compute_dew_point",
    "compute_leaf_wetness_proxy",
    "compute_disease_risk_score",
    "compute_enhanced_disease_risk",
    # state
    "GreenhouseState",
    "ActuatorState",
    "MPCState",
    "WeatherState",
    "GrowthClassificationOutput",
    "DiseaseClassificationOutput",
    "DiseaseProgressionOutput",
    "GrowthProgressionOutput",
    "FusedState",
    "DigitalTwinStepPayload",
    "DigitalTwinTrajectoryPayload",
    "ComparisonMetrics",
    "EvaluationSummary",
    "ImagePayload",
    "ControllerDecisionContext",
    # setpoints
    "StageSetpoint",
    "StageControlProfile",
    "get_setpoint",
    "get_control_profile",
    "get_all_setpoints",
    "get_all_control_profiles",
    # constraints
    "ConstraintSet",
    "get_default_constraints",
    "merge_constraints",
    "tighten_constraints_for_disease",
    # config
    "MPCConfig",
    "load_mpc_config",
    # input preparation
    "MPCInputPreparation",
    # image streaming
    "ImageStreamer",
    # model orchestration
    "WeatherDisturbanceForecast",
    "DiseaseRiskPenalty",
    "GrowthStageWeights",
    "StateFusion",
    "DigitalTwinOutput",
    "MPCRunner",
    # greenhouse model
    "GreenhouseModelParams",
    "GreenhouseTransitionModel",
    # baseline controller
    "RuleBasedController",
    "BaselineControlPayload",
    # cost function
    "DiseaseContext",
    "StageCost",
    "TerminalCost",
    "CostBuilder",
    # weather adaptation
    "WeatherAdaptiveModifiers",
    "compute_weather_adaptation",
    # mpc solver
    "MPCSolver",
    "MPCSolution",
    # explanation
    "ExplanationEntry",
    "ControllerExplanation",
    "ExplanationBuilder",
    # evaluation metrics
    "TrackingMetrics",
    "DiseaseBurdenMetrics",
    "ResourceMetrics",
    "ControlQualityMetrics",
    "SafetyMetrics",
    "ControllerMetricsBundle",
    "compute_all_metrics",
    # utilities
    "discover_latest_artifact",
    # yield proxy
    "YieldProxyWeights",
    "YieldProxyResult",
    "compute_yield_proxy",
    # experiment runner
    "ExperimentConfig",
    "ExperimentRunner",
    "ComparisonReport",
    "make_baseline_adapter",
    "make_mpc_adapter",
    "generate_default_weather",
    "generate_default_growth_stages",
    "make_default_initial_state",
    # evaluation API
    "run_evaluation",
    "save_evaluation_artifacts",
    "load_evaluation_report",
    # digital twin state & interface
    "DTSnapshot",
    "DTStepInput",
    "DTStepOutput",
    "DTDiagnostics",
    "DigitalTwinEngine",
    "DigitalTwinPlant",
    # dt runtime loop
    "prepare_initial_state",
    "prepare_weather_sequence",
    "prepare_growth_stages",
    "build_mpc_solver",
    "build_fused_state",
    "DTLoop",
    "DTLoopStepResult",
    "MPC_CADENCE_STEPS",
    "IMAGE_CADENCE_STEPS",
    "should_force_mpc_update",
    "DTLoopLogger",
    "DTLoopRunSummary",
    # dt input / output / image abstractions
    "DTInputProvider",
    "ImageObservation",
    "SyntheticInputProvider",
    "DatabaseInputProvider",
    "ImageObserver",
    "MinIOImageObserver",
    "SyntheticImageObserver",
    "DTOutputWriter",
    "JsonFileOutputWriter",
    "fanout_step_to_writer",
    # dt artifact manager
    "DTArtifactManager",
    # realtime closed-loop core
    "RealtimeLoop",
    "RealtimeLoopConfig",
    "RealtimeMPCInputPreparation",
    "RealtimeRunSummary",
    "RealtimeStepResult",
    "RunRegistry",
    "STAGE_DURATION_HOURS",
    "PRIOR_STAGE_HOURS",
    "ensure_stream_table",
    "seed_initial_state",
    "write_step_to_stream",
    "estimate_energy",
    "diurnal_temp",
    "diurnal_humidity",
    "diurnal_solar",
    "default_soil_moisture",
    # minio image observer
    "MinIOImageObserver",
]
