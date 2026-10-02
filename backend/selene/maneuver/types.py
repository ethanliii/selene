"""FilterRun re-export with a local fallback (the OD track owns ``selene.od.types``).

The maneuver detectors only need the field names documented in ``selene/od/types.py``; if that
module is absent (e.g. the OD track has not landed yet) an identical dataclass is defined here.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised implicitly
    from selene.od.types import FilterRun  # type: ignore
except ImportError:  # pragma: no cover
    from dataclasses import dataclass, field

    import numpy as np

    @dataclass
    class FilterRun:  # type: ignore[no-redef]
        """See ``selene.od.types.FilterRun`` (identical fields)."""

        t_s: np.ndarray
        x: np.ndarray
        P: np.ndarray
        x_pred: np.ndarray
        P_pred: np.ndarray
        innov: np.ndarray
        S: np.ndarray
        nis: np.ndarray
        meas: list
        object_id: str
        meta: dict = field(default_factory=dict)

        def __len__(self) -> int:
            return int(np.asarray(self.t_s).size)


__all__ = ["FilterRun"]
