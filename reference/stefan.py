"""Neumann solution to the one-dimensional two-phase Stefan problem.

Setup: a semi-infinite solid occupies z > 0, initially at uniform temperature
T_i below the melt temperature T_m. At t = 0 the wall z = 0 is raised to and
held at T_0 > T_m. A melt front s(t) propagates into the solid:

    s(t) = 2 * lam * sqrt(alpha * t)

where alpha = k / (rho * c) is the thermal diffusivity and lam solves the
transcendental equation (equal thermophysical properties assumed in both
phases, which is what the GaussianBench melt scenes declare):

    St_l / (exp(lam^2) * erf(lam))
      - St_s / (exp(lam^2) * erfc(lam)) = lam * sqrt(pi)

with the liquid and solid Stefan numbers

    St_l = c * (T_0 - T_m) / L_f      (superheat driving melting)
    St_s = c * (T_m - T_i) / L_f      (subcooling resisting melting)

and L_f the latent heat of fusion.

Derivation and general form (unequal properties): H. S. Carslaw and
J. C. Jaeger, "Conduction of Heat in Solids", 2nd ed., Oxford, 1959,
chapter 11 (the Neumann problem), or V. Alexiades and A. D. Solomon,
"Mathematical Modeling of Melting and Freezing Processes", Hemisphere, 1993,
chapter 2. The equal-properties simplification collapses their two
diffusivities into one alpha.

Regime assumptions: one-dimensional conduction, constant properties, sharp
front, no convection in the melt, no volume change on melting.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc


def neumann_lambda(stefan_liquid: float, stefan_solid: float) -> float:
    """Solve the Neumann transcendental equation for lam.

    The left side decreases monotonically from +inf (lam -> 0, the erf term
    blows up) to -inf (large lam, the erfc term blows up with a minus sign),
    so a bracketing solve is robust.
    """
    if stefan_liquid <= 0:
        raise ValueError("stefan_liquid must be positive (T_0 > T_m).")
    if stefan_solid < 0:
        raise ValueError("stefan_solid must be nonnegative (T_i <= T_m).")

    def f(lam: float) -> float:
        e = np.exp(lam * lam)
        term_l = stefan_liquid / (e * erf(lam))
        term_s = (stefan_solid / (e * erfc(lam))) if stefan_solid > 0 else 0.0
        return term_l - term_s - lam * np.sqrt(np.pi)

    lo, hi = 1e-9, 1.0
    while f(hi) > 0:
        hi *= 2.0
        if hi > 50.0:
            raise RuntimeError("Neumann lambda bracket search failed.")
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))


def front_prefactor(conductivity_w_m_k: float, density_kg_m3: float,
                    specific_heat_j_kg_k: float, latent_heat_j_kg: float,
                    wall_temperature_k: float, melt_temperature_k: float,
                    initial_temperature_k: float) -> float:
    """Prefactor a in s(t) = a * sqrt(t), in m / sqrt(s): a = 2 lam sqrt(alpha)."""
    alpha = conductivity_w_m_k / (density_kg_m3 * specific_heat_j_kg_k)
    st_l = specific_heat_j_kg_k * (wall_temperature_k - melt_temperature_k) \
        / latent_heat_j_kg
    st_s = specific_heat_j_kg_k * (melt_temperature_k - initial_temperature_k) \
        / latent_heat_j_kg
    lam = neumann_lambda(st_l, st_s)
    return 2.0 * lam * np.sqrt(alpha)
