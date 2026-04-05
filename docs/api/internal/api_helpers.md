# API Helpers (Internal)

These functions support the public API by handling **input validation, tolerance
management, and symmetrization workflows**.

They act as a bridge between user-facing functions and the core symmetry engine.

---

## Overview

The API layer is responsible for:

* Validating user inputs
* Preparing numerical tolerances
* Applying symmetry operations to molecular geometry
* Evaluating candidate symmetrizations

---

## Notes

* These functions are **internal to the API layer** and not intended for direct use
* They rely heavily on internal core structures
* Changes here may affect public API behavior but not the underlying symmetry engine

---

::: minimalsym.api
    options:
      filters:
      - "^_"