"""
Shared numerical constants for the symmetrizer package.

Centralizing these values avoids scattered magic numbers and makes it easy
to tune tolerances globally without hunting through multiple files.
"""

# ── Numerical tolerances ──────────────────────────────────────────────────────

# Internal floating-point comparison tolerance used across mol_ops and sym_ops.
# Governs axis collinearity checks, zero-vector detection, etc.
NUMERICAL_TOL: float = 1e-8

# Tolerance stamped onto the molecule returned by ``symmetrize()``.
# After symmetrization the geometry is exact to machine precision, so a very
# tight value is appropriate here.
SYMMETRIZED_TOL: float = 1e-12

# ── Icosahedral geometry constants ───────────────────────────────────────────

# Angle (radians) between a C2 axis and the nearest C3 axis in an icosahedron.
# Derived from the golden ratio phi = (1+sqrt(5))/2:
#   theta = arccos(phi / sqrt(1 + phi^2)) ~= 0.3649 rad ~= 20.9 deg
# Used in find_point_group to identify the secondary axis for Ih/I molecules.
IH_C2_C3_ANGLE: float = 0.36486382647383764

# Angular tolerance for the IH_C2_C3_ANGLE comparison.
IH_ANGLE_TOL: float = 1e-4
