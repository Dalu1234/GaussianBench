"""Analytical beam references, pure numpy.

Formulas and regime assumptions:

Euler-Bernoulli cantilever fundamental frequency:
    f1 = (beta1^2 / (2*pi*L^2)) * sqrt(E*I / (rho*A))
where beta1*L = 1.875104 is the first root of cosh(x)*cos(x) = -1,
I is the second moment of area about the bending axis, A the cross section.
Assumes: slender beam (shear and rotary inertia negligible, Timoshenko
corrections small for L/h >= 8), small deflection, linear elastic, uniform
section. Young's modulus enters directly (uniaxial stress; lateral faces
free). Citation: S. S. Rao, "Mechanical Vibrations", 5th ed., section 8.5,
or any structural dynamics text.
"""

from __future__ import annotations

import numpy as np

BETA1_L = 1.875104068711961  # first root of cosh(x) cos(x) = -1


def cantilever_f1(youngs_modulus_pa: float, density_kg_m3: float,
                  length_m: float, width_m: float, height_m: float) -> float:
    """Fundamental bending frequency in Hz for a rectangular cantilever.

    Bending is across height_m (the direction of the pluck); width_m is the
    other lateral dimension. I = w*h^3/12, A = w*h.
    """
    I = width_m * height_m ** 3 / 12.0
    A = width_m * height_m
    return (BETA1_L ** 2 / (2.0 * np.pi * length_m ** 2)
            * np.sqrt(youngs_modulus_pa * I / (density_kg_m3 * A)))


def mode1_shape(s: np.ndarray) -> np.ndarray:
    """First cantilever mode shape phi1(s) for s = x/L in [0, 1].

    phi1(s) = cosh(b s) - cos(b s) - sigma (sinh(b s) - sin(b s)),
    sigma = (cosh b - cos b) / (sinh b + sin b), b = beta1*L root.
    Normalized so phi1(1) = 1 by the caller if needed.
    """
    b = BETA1_L
    sigma = (np.cosh(b) - np.cos(b)) / (np.sinh(b) + np.sin(b))
    return (np.cosh(b * s) - np.cos(b * s)
            - sigma * (np.sinh(b * s) - np.sin(b * s)))
