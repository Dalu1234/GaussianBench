"""Yeoh hyperelastic reference for natural rubber in uniaxial tension.

This is the first MEASURED-data reference in the suite (model_validation),
as opposed to the analytical references elsewhere (verification).

Material identity: a natural-rubber compound characterized by a uniaxial
tensile test per ASTM D412 (10 mm/min, ambient temperature) and fit to a
3-term Yeoh strain-energy function in

  O. Azarniya, G. H. Rahimi, "Numerical and experimental analysis of free
  vibrations and static bending of a sandwich beam with a hyperelastic
  core," Mechanics Based Design of Structures and Machines, 2022,
  doi:10.1080/15397734.2022.2121721.
  (Coefficients: Table 2. Their Fig 9 shows the Yeoh fit tracks the raw
  measured stress-strain curve, so it faithfully stands in for the
  measurement.)

  W = C10 (I1 - 3) + C20 (I1 - 3)^2 + C30 (I1 - 3)^3

For INCOMPRESSIBLE uniaxial tension with stretch lambda (transverse stretch
1/sqrt(lambda)), the Cauchy stress along the load direction is

  sigma(lambda) = 2 (lambda^2 - 1/lambda)
                  * [C10 + 2 C20 (I1 - 3) + 3 C30 (I1 - 3)^2],
  I1 = lambda^2 + 2/lambda.

Regime: incompressible, quasi-static, isothermal. Real rubber is mildly
rate-dependent (viscoelastic); the source reports tensile scatter under 6
percent, which the model_validation tolerance budgets for.
"""

import numpy as np

# Yeoh coefficients (Pa), Azarniya & Rahimi 2022, Table 2.
C10 = 708663.01
C20 = 19707.6
C30 = -149.15

# Initial shear and Young's moduli (small-strain limit).
INITIAL_SHEAR_MODULUS = 2.0 * C10          # mu = 2 C10 = 1.417 MPa
INITIAL_YOUNGS_MODULUS = 6.0 * C10         # E = 3 mu = 6 C10 = 4.25 MPa (incompr.)

# Density used in the source study's parametric setup (kg/m^3). Not a
# specimen-measured value; irrelevant to a quasi-static stress-strain test.
DENSITY_KG_M3 = 1100.0


def uniaxial_cauchy_stress(stretch):
    """Cauchy stress (Pa) in the load direction for incompressible uniaxial
    tension at the given stretch(es) lambda >= 1."""
    lam = np.asarray(stretch, dtype=float)
    I1 = lam ** 2 + 2.0 / lam
    Wp = C10 + 2.0 * C20 * (I1 - 3.0) + 3.0 * C30 * (I1 - 3.0) ** 2
    return 2.0 * (lam ** 2 - 1.0 / lam) * Wp
