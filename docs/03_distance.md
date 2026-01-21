# Distance Module

This document describes the distance and similarity computations used in the VecDB MVP,
including L2 (squared) distance, cosine similarity, and vector normalization.

The Distance module provides the mathematical foundation for all search algorithms
(brute-force baseline and HNSW).

---

## Supported Metrics

### 1) Squared L2 Distance

Given two vectors `a` and `b` of dimension `d`:

L2 distance:
‖a − b‖₂ = sqrt( Σ (aᵢ − bᵢ)² )

Squared L2 distance (used in VecDB):
Σ (aᵢ − bᵢ)²

#### Why squared L2?
- Preserves ordering: if dist₁ < dist₂, then sqrt(dist₁) < sqrt(dist₂)
- Avoids expensive `sqrt` in inner loops
- Common practice in ANN systems

**Interpretation**
- Smaller value → vectors are closer
- Zero → identical vectors

---

### 2) Dot Product

Dot product of vectors `a` and `b`:

a · b = Σ (aᵢ · bᵢ)

Used as a building block for cosine similarity.

---

### 3) Cosine Similarity & Cosine Distance

Cosine similarity measures the angle between two vectors:

cos(a, b) = (a · b) / (‖a‖ · ‖b‖)

To unify comparison logic with L2 (smaller is better),
VecDB uses **cosine distance**:

cosine_distance(a, b) = 1 − cos(a, b)

#### Properties
- Range: [0, 2]
- 0 → same direction
- 1 → orthogonal
- 2 → opposite direction

---

## Vector Normalization

### L2 Normalization

Given vector `v`, normalization computes:

v̂ = v / ‖v‖

After normalization:
‖v̂‖ = 1

#### Why normalize?
- If all vectors are unit-length, cosine similarity reduces to dot product:
  cos(a, b) = a · b
- Eliminates per-query norm computation
- Improves performance and numerical stability

In VecDB MVP, normalization is performed **explicitly and in-place**,
allowing the caller to choose whether and when to normalize.

---

## API Overview

### `Distance::distance(metric, a, b, dim)`

**Purpose**
- Unified interface for computing distance between two vectors.

**Inputs**
- `metric`: L2 or COSINE
- `a`, `b`: pointers to float arrays
- `dim`: vector dimension

**Output**
- Distance value where **smaller is closer**

**Notes**
- L2 returns squared distance
- COSINE returns `1 - cosine_similarity`

---

### `Distance::normalize_inplace(v, dim)`

**Purpose**
- Normalize a vector to unit length.

**Behavior**
- Modifies the input vector in-place
- No-op if vector norm is zero

---

## Design Rationale Summary

- All metrics are expressed as “distance” (lower is better)
- Squared L2 avoids unnecessary computation
- Normalization enables efficient cosine search
- Metric abstraction keeps search code independent of math details

---

## Role in the System

- Used by:
  - Bruteforce baseline (exact search)
  - HNSW index (approximate search)
- Guarantees consistent ordering across different algorithms
