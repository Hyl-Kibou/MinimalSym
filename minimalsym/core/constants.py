"""
Shared numerical constants for the core package.
"""

# ── Numerical tolerances ──────────────────────────────────────────────────────

NUMERICAL_TOL: float = 1e-8
"""
Internal floating-point comparison tolerance used across mol_ops and sym_ops.
Governs axis collinearity checks, zero-vector detection, etc.
"""

# ── Icosahedral geometry constants ───────────────────────────────────────────

IH_C2_C3_ANGLE: float = 0.36486382647383764
"""
Angle (radians) between a C2 axis and the nearest C3 axis in an icosahedron.
Derived from the golden ratio phi = (1+sqrt(5))/2:
    theta = arccos(phi^2 / sqrt(1 + phi^4)) = arctan(1/phi^2) ~= 0.3649 rad ~= 20.9 deg
Used in find_point_group to identify the secondary axis for Ih/I molecules.
"""

IH_ANGLE_TOL: float = 1e-4
"""
Angular tolerance for the IH_C2_C3_ANGLE comparison.
"""

# ── Warning Control ──────────────────────────────────────────────────────────

PRINT_WARNINGS: bool = False
"""
If `True` warnings and debug messages are printed.
"""