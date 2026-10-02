"""Linear stability of CR3BP periodic orbits from the monodromy matrix.

The monodromy matrix M = Φ(T, 0) is the state transition matrix over one period.  Its six
eigenvalues come in reciprocal pairs (λ, 1/λ) because the CR3BP is Hamiltonian and
time-reversible; one pair is (1, 1) for an autonomous periodic orbit (phase direction and the
family tangent).  The stability index used here follows Howell (1984) / Zimovan-Spreen et al.
(2020):

    ν = ½ (|λ_max| + 1/|λ_max|),

where λ_max is the eigenvalue of largest magnitude.  ν = 1 for a linearly stable orbit (all
eigenvalues on the unit circle); ν ≫ 1 means strong hyperbolic instability with e-folding
time T / ln|λ_max|.  JPL's Three-Body Periodic Orbit catalogue reports the same quantity
("stability index ≤ 1 ⇒ stable").  We also return Broucke's per-pair indices
s_i = λ_i + 1/λ_i (real for real or unit-circle pairs).

Units: nondimensional CR3BP (``selene.dynamics.cr3bp``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from selene.constants import MU
from selene.dynamics.cr3bp import propagate_cr3bp

__all__ = ["StabilityResult", "monodromy", "stability_from_monodromy", "stability_index"]

STABLE_TOL = 1e-6


@dataclass
class StabilityResult:
    eigenvalues: np.ndarray      # complex (6,), sorted by decreasing magnitude
    nu: float                    # ½(|λ_max| + 1/|λ_max|)
    stable: bool                 # nu <= 1 + STABLE_TOL
    det: float                   # det(M); should be 1 (symplectic)
    unit_pair_error: float       # |λ - 1| for the eigenvalue closest to +1 (should be ~0)
    pair_indices: np.ndarray     # Broucke indices s_i = λ_i + 1/λ_i, (3,) complex
    monodromy: np.ndarray        # (6,6)

    @property
    def abs_eigenvalues(self) -> np.ndarray:
        return np.abs(self.eigenvalues)

    @property
    def efold_time(self) -> float:
        """e-folding time of the dominant unstable mode, in periods (inf if stable)."""
        lmax = np.max(np.abs(self.eigenvalues))
        return float("inf") if lmax <= 1.0 + STABLE_TOL else 1.0 / np.log(lmax)


def monodromy(ic, period: float, mu: float = MU, rtol: float = 1e-12, atol: float = 1e-12, t_eval=None):
    """Integrate state + STM over one period.  Returns ``(M (6,6), sol)`` where ``sol`` is the
    scipy result (``sol.y`` has 42 rows at ``t_eval`` if given, else at the end point only)."""
    ic = np.asarray(ic, dtype=np.float64).ravel()[:6]
    if t_eval is None:
        sol = propagate_cr3bp(ic, tf=float(period), mu=mu, stm=True, rtol=rtol, atol=atol)
    else:
        sol = propagate_cr3bp(ic, t_eval=np.asarray(t_eval, dtype=np.float64), mu=mu, stm=True, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"monodromy integration failed: {sol.message}")
    M = sol.y[6:, -1].reshape(6, 6)
    return M, sol


def stability_from_monodromy(M: np.ndarray) -> StabilityResult:
    M = np.asarray(M, dtype=np.float64).reshape(6, 6)
    lam = np.linalg.eigvals(M)
    order = np.argsort(-np.abs(lam))
    lam = lam[order]
    lmax = float(np.abs(lam[0]))
    nu = 0.5 * (lmax + 1.0 / lmax)
    # pair indices: pair each eigenvalue with the one closest to its reciprocal
    used = np.zeros(6, bool)
    pairs = []
    for i in range(6):
        if used[i]:
            continue
        used[i] = True
        target = 1.0 / lam[i] if lam[i] != 0 else np.inf
        cand = [j for j in range(6) if not used[j]]
        if not cand:
            break
        j = min(cand, key=lambda k: abs(lam[k] - target))
        used[j] = True
        pairs.append(lam[i] + lam[j])
    while len(pairs) < 3:
        pairs.append(np.nan + 0j)
    unit_err = float(np.min(np.abs(lam - 1.0)))
    return StabilityResult(
        eigenvalues=lam,
        nu=float(nu),
        stable=bool(nu <= 1.0 + STABLE_TOL),
        det=float(np.linalg.det(M)),
        unit_pair_error=unit_err,
        pair_indices=np.array(pairs[:3]),
        monodromy=M,
    )


def stability_index(ic, period: float, mu: float = MU) -> StabilityResult:
    """Monodromy eigen-analysis of the periodic orbit ``(ic, period)``."""
    M, _ = monodromy(ic, period, mu)
    return stability_from_monodromy(M)
