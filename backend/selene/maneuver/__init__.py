"""Maneuver detection and impulsive-Δv estimation (PLAN §2.7, milestone M5).

Public surface:

* :mod:`selene.maneuver.config` — :class:`DetectorConfig`, :class:`EstimatorConfig`, :class:`FilterConfig`
* :mod:`selene.maneuver.detection` — filter-health gate, NIS / windowed NIS / gap re-fit Mahalanobis / NEES / CUSUM tests, :func:`detect`
* :mod:`selene.maneuver.estimation` — :func:`estimate_impulsive_dv`, :func:`classify_maneuver`
* :mod:`selene.maneuver.synthetic` — truth with injected burn, measurement generation (SIMULATED)
* :mod:`selene.maneuver.pipeline` — :func:`run_scenario` end-to-end runner used by the API
* :mod:`selene.maneuver._simple_ekf` — compact STM-based angles-only EKF returning the shared FilterRun
"""
from selene.maneuver.config import DetectorConfig, EstimatorConfig, FilterConfig  # noqa: F401
from selene.maneuver.detection import STATUS_VALUES, Detection, DetectionReport, detect  # noqa: F401
from selene.maneuver.estimation import DvEstimate, classify_maneuver, estimate_impulsive_dv  # noqa: F401
from selene.maneuver.types import FilterRun  # noqa: F401

__all__ = [
    "DetectorConfig", "EstimatorConfig", "FilterConfig", "Detection", "DetectionReport", "detect", "STATUS_VALUES",
    "DvEstimate", "classify_maneuver", "estimate_impulsive_dv", "FilterRun",
]
