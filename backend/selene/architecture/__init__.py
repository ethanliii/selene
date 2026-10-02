"""Architecture trade studio: candidate sensor platforms and Monte Carlo scoring (M8)."""
from selene.architecture.candidates import (  # noqa: F401
    PLATFORMS,
    PRESET_ARCHITECTURES,
    Architecture,
    SensorSpec,
    architecture_from_dict,
    preset_architectures,
)
from selene.architecture.montecarlo import EvalConfig, EvaluationResult, ManeuverModel, burn_displacement, evaluate  # noqa: F401

__all__ = [
    "PLATFORMS", "PRESET_ARCHITECTURES", "Architecture", "SensorSpec", "architecture_from_dict", "preset_architectures",
    "EvalConfig", "EvaluationResult", "ManeuverModel", "burn_displacement", "evaluate",
]
