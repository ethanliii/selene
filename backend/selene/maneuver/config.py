"""Configuration dataclasses for maneuver detection, Δv estimation and the synthetic filter.

All probabilities are per-test false-alarm rates (α); the detectors in
:mod:`selene.maneuver.detection` convert them to χ² thresholds.  Units: hours/seconds as
named, km, km/s, m/s where the field name says ``_mps``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

__all__ = ["DetectorConfig", "EstimatorConfig", "FilterConfig"]


@dataclass
class DetectorConfig:
    """Thresholds and windows for the detection tests.

    alpha, alpha_nis, alpha_window, alpha_gap
        Per-test false-alarm probabilities.  ``alpha`` is the default; the three optional
        fields override it for the per-update NIS, the windowed NIS and the gap re-fit test
        respectively (``None`` → ``alpha``).  For the per-update NIS test α is the probability
        that a *single* consistent update exceeds χ²_m(1−α); the run-level declaration uses a
        Šidák family-wise correction (``declare_familywise``) so that one run of N tested
        updates has ≈ α probability of a false *declaration*.
    window
        Number of consecutive updates summed in the windowed NIS test (Σ NIS ~ χ²_{m·window}).
    gap_hours_for_refit, gap_cadence_factor
        An observation gap triggers the post-gap short-arc re-fit / Mahalanobis test when it is
        longer than ``gap_hours_for_refit`` (hours) **and** longer than ``gap_cadence_factor`` ×
        the median inter-observation interval of the run.  The second condition makes "gap"
        mean *an outage relative to the track's normal cadence* (lunar glare, daylight); at a
        24-h cadence every interval exceeds 12 h, but none is a gap in that sense.
    refit_n_obs, refit_min_obs
        Number of post-gap observations used in the short-arc batch fit and the minimum needed
        to attempt it.  3 angle pairs give 6 equations for 6 unknowns (an exact fit); more
        leave 2k−6 degrees of freedom with which the arc's *internal* consistency is checked —
        a burn inside the arc (rather than inside the gap) makes the arc fit inconsistent, and
        such a gap test is skipped instead of being attributed to the gap.  Measured on
        SIM-DRO-01 (1″, 6-h cadence, one observer; angles-only range freedom lets a few
        points absorb a kink): a 5 m/s burn between the 2nd and 3rd post-gap observations
        gives arc χ² 4.8/2 dof with 4 obs (undetectable), 264/6 with 6, 868/10 with 8; a
        1 m/s burn 4.6, 13.7 and 33.9 (threshold 23.2) — hence the default of 8, while a
        burn inside the gap and a quiet arc stay consistent (8.6/10 and 9.2/10).  With fewer
        than ~6 available post-gap observations the check has little power (``arc_dof`` is
        reported with every gap detection).
    refit_prior_inflation
        The short-arc fit is regularised with a weak prior = ``refit_prior_inflation × P_pred``
        so that an unobservable range direction cannot make the normal matrix singular.  The
        prior makes the test slightly conservative (documented in ``gap_refit_test``).
    baseline_updates, baseline_alpha
        Filter-health gate (no truth needed).  Maneuver tests start only after a *baseline* of
        ``baseline_updates`` consecutive updates whose summed NIS passes χ²_{m·W}(1−baseline_alpha),
        i.e. after the filter has demonstrated it is tracking.  A filter that never converges
        (bad prior, strongly nonlinear regime, model mismatch) therefore yields the status
        ``filter_not_converged`` instead of a spurious maneuver declaration.  Set to 0 to
        disable.  Limitation: a burn *during* the baseline window is indistinguishable from a
        bad prior and is reported as non-convergence, not as a maneuver.
    nees_alpha
        Two-sided significance for the NEES filter-consistency monitor (needs truth).
    nees_health_alpha, nees_health_max_ratio, nees_health_gates
        Simulation-only health check on the mean NEES over the baseline epochs (truth needed).
        Above the statistical bound χ²_{6n}(1−nees_health_alpha)/n the covariance is flagged
        *optimistic* (warning only: a mildly optimistic UKF still localises a burn correctly);
        above ``nees_health_max_ratio`` × 6 (default 10× the χ²₆ mean, i.e. errors ≳ 3σ on
        average) the filter is *grossly* inconsistent and, when ``nees_health_gates`` is True,
        the run is not declared (status ``filter_inconsistent``).  Operationally no truth
        exists, so only the NIS baseline gate applies there.
    cusum_k, cusum_arl0, cusum_h
        One-sided CUSUM on (NIS − m): allowance ``k`` (in NIS units), target in-control average
        run length ARL₀ (updates per false alarm) used to tune the decision threshold ``h`` by
        Monte Carlo when ``cusum_h`` is None.
    min_updates_before_test
        Skip the first updates (initialisation transient) before the baseline search.
    m
        Measurement dimension (2 for RA/Dec).
    """

    alpha: float = 0.01
    alpha_nis: Optional[float] = None
    alpha_window: Optional[float] = None
    alpha_gap: Optional[float] = None
    window: int = 5
    gap_hours_for_refit: float = 12.0
    gap_cadence_factor: float = 2.5
    refit_n_obs: int = 8
    refit_min_obs: int = 4
    refit_prior_inflation: float = 1e4
    baseline_updates: int = 5
    baseline_alpha: float = 0.05
    nees_alpha: float = 0.05
    nees_health_alpha: float = 1e-3
    nees_health_max_ratio: float = 10.0
    nees_health_gates: bool = True
    cusum_enabled: bool = True
    cusum_k: float = 1.0
    cusum_arl0: float = 200.0
    cusum_h: Optional[float] = None
    #: CUSUM alarms are always *reported*; they only count toward the run-level declaration when
    #: True, because the CUSUM's false-alarm rate is 1/ARL₀ per update (≈ 12 % over a 26-update
    #: run at ARL₀ = 200), not a family-wise α.
    cusum_declares: bool = False
    min_updates_before_test: int = 2
    declare_familywise: bool = True
    m: int = 2

    def __post_init__(self):
        for name in ("alpha", "alpha_nis", "alpha_window", "alpha_gap", "baseline_alpha", "nees_alpha", "nees_health_alpha"):
            v = getattr(self, name)
            if v is not None and not (0.0 < v < 1.0):
                raise ValueError(f"{name} must be in (0, 1)")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.refit_min_obs < 3:
            raise ValueError("refit_min_obs must be >= 3 (6 equations for 6 unknowns)")
        if self.refit_n_obs < self.refit_min_obs:
            raise ValueError("refit_n_obs must be >= refit_min_obs")
        if self.baseline_updates < 0:
            raise ValueError("baseline_updates must be >= 0 (0 disables the gate)")

    # resolved per-test alphas -----------------------------------------------------------
    @property
    def a_nis(self) -> float:
        return self.alpha if self.alpha_nis is None else self.alpha_nis

    @property
    def a_window(self) -> float:
        return self.alpha if self.alpha_window is None else self.alpha_window

    @property
    def a_gap(self) -> float:
        return self.alpha if self.alpha_gap is None else self.alpha_gap


@dataclass
class EstimatorConfig:
    """Impulsive-Δv estimator settings (:func:`selene.maneuver.estimation.estimate_impulsive_dv`).

    n_grid, grid_rtol, grid_max_nfev
        Candidate burn epochs placed at the centres of ``n_grid`` equal sub-intervals of the
        detection window, each solved coarsely (``grid_rtol``, ``grid_max_nfev``: enough to rank
        them); the best is re-solved at full precision and then refined continuously (``refine``).
    n_post_obs
        Maximum number of post-window observations used in the fit.
    include_prior
        Solve for the pre-burn state correction as well (6 extra unknowns with the filter's
        covariance as prior) so the Δv covariance includes pre-burn state uncertainty.
    dv_bound_kms
        Box bound on each Δv component (km/s) for the trust-region solver.
    center
        Frame centre for the RTN/VNB decomposition: 'earth', 'moon' or 'auto' (Moon when the
        object is closer than ``moon_primary_radius_km`` to the Moon, else Earth).
    """

    n_grid: int = 9
    #: grid-stage (ranking only) integrator tolerance and iteration cap; the best grid point is
    #: re-solved at full precision before the continuous refinement
    grid_rtol: float = 1e-8
    grid_max_nfev: int = 8
    refine: bool = True
    n_post_obs: int = 10
    include_prior: bool = True
    dv_bound_kms: float = 1.0
    max_nfev: int = 60
    center: Literal["auto", "earth", "moon"] = "auto"
    moon_primary_radius_km: float = 100_000.0
    rtol: float = 1e-10
    atol: float = 1e-10


@dataclass
class FilterConfig:
    """Initial covariance and process noise of the sequential filter in the synthetic pipeline.

    sigma_pos0_km, sigma_vel0_mps
        1-σ of the (isotropic) initial state error; the initial estimate is the truth plus a
        draw from this covariance so the filter starts statistically consistent.  ``None``
        (default) selects a regime-dependent *scenario default* (not a physical claim):

        * cislunar / xGEO objects (DROs, halos, NRHO apolune, resonant tours):
          ``(20 km, 2 m/s)`` — the sparse-angles track prior used for the Pfa / NEES
          calibration on SIM-DRO-01 in ``tests/test_maneuver_detection.py``;
        * Moon-bound orbiters (negative Moon-relative two-body energy and within
          ``moon_bound_radius_km`` of the Moon, e.g. the ELFO): ``(2 km, 0.1 m/s)``.  A lunar
          orbiter under custody has a far tighter prior from any previous OD, and — measured in
          this project — the reference filters are only consistent there for a prior this
          tight at a 6-h cadence (UKF mean NEES ≈ 6 at (2 km, 0.1 m/s) vs ≈ 90 at (20 km,
          2 m/s); the STM EKF is inconsistent at every prior tried because the regime stretches
          a 1 km error 12.6× per 6 h and the linearised gain collapses the unobserved range
          direction).  With an explicit prior the filter-health gate in
          :class:`DetectorConfig` still protects the maneuver verdict.
    q_psd_km2_s3
        White acceleration PSD q [km²/s³] feeding Q = q·[[Δt³/3 I, Δt²/2 I], [Δt²/2 I, Δt I]].
        Default 1e-18 ≈ (10 % of the SRP acceleration of a C_R·A/m = 0.01 m²/kg bus, 5e-12
        km/s²)² × 1-day correlation time, i.e. 0.15 mm/s and 1.8 m of growth per 6 h.  The
        synthetic truth uses the same force model as the filter, so this is effectively a
        numerical floor; the NEES of the reference EKF is ≈ 5.8 (χ²₆ mean 6) at this value
        and drops to ≈ 3.8 (over-conservative) at 1e-16 — see ``tests/test_maneuver_detection.py``.
    """

    sigma_pos0_km: Optional[float] = None
    sigma_vel0_mps: Optional[float] = None
    q_psd_km2_s3: float = 1e-18
    rtol: float = 1e-10
    atol: float = 1e-10
    cislunar_prior: tuple[float, float] = (20.0, 2.0)
    moon_bound_prior: tuple[float, float] = (2.0, 0.1)
    moon_bound_radius_km: float = 20_000.0
