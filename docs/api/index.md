# API Reference

The MolSymPy API provides tools for **molecular symmetry detection,
symmetrization, and symmetry exploration** based on point group theory.

---

## Overview

The API is organized into two layers:

### Public API

High-level, stable functions for everyday use:

* Symmetrize molecular geometries
* Detect point groups
* Check planarity
* Identify symmetry-inequivalent atoms
* Generate alternative symmetry-consistent structures

See: [Public API](public.md)

---

### Internal API

Lower-level implementation used by the symmetry engine.

These functions:

* Implement symmetry detection algorithms
* Construct symmetry elements and point groups
* Manipulate molecular geometry
* Generate and rank symmetry candidates

!!! warning 
    Internal APIs are **not stable** and may change between versions.

See: [Internal Documentation](internal/overview.md)

---

## How the API Works

At a high level, MolSymPy follows this pipeline:

```
Atoms
  ↓
Validation & tolerance setup
  ↓
Point group detection (pg_detect)
  ↓
Symmetry element generation (symel_gen)
  ↓
Symmetry representation (Symtext)
  ↓
Symmetrization / candidate generation
  ↓
Atoms
```

Key idea:

* Detection determines *what symmetry exists*
* Representation encodes *how symmetry acts*
* Geometry applies *how symmetry modifies coordinates*

This pipeline combines:

* **Geometric analysis** (moments of inertia, distances)
* **Group theory** (point groups, symmetry elements)
* **Numerical validation** (tolerances, invariance checks)

For a detailed breakdown, see:
[Data Flow](../flowchart.md)

---

## Choosing the Right Entry Point

* Use the **Public API** if you want:

    * Reliable, high-level functionality
    * Stable interfaces

* Explore the **Internal API** if you want:

    * Algorithmic details
    * Custom extensions
    * Performance tuning

---

## Notes

* Public functions are defined in `api.py`
* Internal modules live under `molsympy/core`
* Functions prefixed with `_` are internal by convention

---

## Next Steps

* See [Public API](public.md)
* Explore internal modules by concept (detection, geometry, decomposition)
* Review the [Data Flow](../flowchart.md) for a system-level view
