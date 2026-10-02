"""SELENE sensor tasking (Milestone 7): linear-covariance custody scheduling for many objects.

* :mod:`~selene.tasking.information` -- STM/process-noise covariance bookkeeping, angles-only
  linearised update, acquisition probability, expected gain (log-det / trace / max-eigenvalue).
* :mod:`~selene.tasking.greedy` -- slot engine + greedy information-theoretic policy.
* :mod:`~selene.tasking.optimize` -- receding-horizon MILP (``scipy.optimize.milp``) with
  local-search fallback and an honestly labelled surrogate objective.
* :mod:`~selene.tasking.metrics` -- custody %, time since last observation, utilisation,
  random and round-robin baselines.
* :mod:`~selene.tasking.scenario` -- object/sensor selection and :class:`Scenario` construction.
"""
from selene.tasking.greedy import Scenario, ScheduleResult, run_greedy, run_schedule  # noqa: F401
from selene.tasking.metrics import custody_metrics, comparison_row, null_custody_pct, run_null, run_random, run_round_robin  # noqa: F401
from selene.tasking.optimize import run_milp  # noqa: F401
from selene.tasking.scenario import make_scenario  # noqa: F401
