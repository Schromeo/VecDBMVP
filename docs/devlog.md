# Development Log — VecDB MVP

This log records the incremental development of the VecDB MVP project,
including environment setup, design decisions, implementation steps,
and future plans. Each entry corresponds to concrete engineering progress
and aligns with the Git commit history.

---

## Backfill — Project Planning & Design Phase (Pre-Implementation)

**Summary**
- Defined the goal of building a Minimal Viable Vector Database (VecDB) from scratch.
- Clarified scope: correctness-first MVP before performance, concurrency, or distribution.
- Studied core concepts behind vector similarity search and HNSW (approximate nearest neighbors).

**Key Design Principles**
- MVP first, extensions later.
- Separate data plane (vector storage) from control plane (index).
- Use brute-force baseline to validate correctness and recall.
- Prefer simple, explainable designs over premature optimization.

**Initial Decisions**
- Language: C++17 (systems-level control, performance, interview relevance).
- Build system: CMake (cross-platform: Windows + macOS).
- IDE: VSCode.
- Version control: Git, with incremental commits.
- Metrics: L2 (squared) and cosine distance.

---

## 2026-01-18 — Project Scaffold & Cross-Platform Build Setup

**Done**
- Initialized GitHub repository `VecDBMVP`.
- Created cross-platform CMake project scaffold.
- Added minimal `main.cpp` to verify build and runtime environment.
- Successfully built and ran on Windows using MSVC.
- Verified Git workflow with clean commit history.

**Why**
- Establish a stable, reproducible build system before writing core logic.
- Ensure portability across Windows and macOS from day one.
- Avoid IDE- or platform-specific assumptions.

**Decisions / Trade-offs**
- Use plain CMake targets instead of IDE-specific project files.
- Delay dependency management (no external libraries in MVP).

**Next**
- Implement distance/metric abstraction.

---

## 2026-01-18 — Distance Module (L2 / Cosine / Normalization)

**Done**
- Implemented `Distance` module with:
  - Squared L2 distance (`L2^2`)
  - Dot product
  - Vector L2 norm
  - In-place normalization
  - Cosine similarity and cosine distance (`1 - cosine_similarity`)
- Unified all metrics under a single `distance(metric, a, b)` interface.
- Added sanity tests in `main.cpp` to validate numeric correctness.

**Why**
- Distance computation is a core dependency for both brute-force baseline
  and HNSW index.
- Squared L2 avoids unnecessary `sqrt` while preserving ordering.
- Unified API simplifies downstream search logic.

**Decisions / Trade-offs**
- Use squared L2 instead of true L2 for performance.
- Represent cosine as a distance (lower is better) to unify comparison logic.
- Support vector normalization to enable cosine similarity via dot product.

**Verification**
- Manually validated expected outputs:
  - L2^2([1,0],[2,0]) = 1
  - Cosine distance aligned with geometric intuition.
  - Normalization produced unit vectors.

**Next**
- Implement vector storage layer.

---

## 2026-01-19 — Environment Migration: Windows → macOS

**Done**
- Installed macOS development toolchain:
  - Apple Clang (via Xcode Command Line Tools)
  - CMake (via Homebrew)
- Cloned existing GitHub repository to macOS.
- Configured CMake + VSCode successfully.
- Resolved generator differences (single-config Unix Makefiles vs MSVC multi-config).
- Verified executable runs correctly on macOS.

**Why**
- Ensure project remains fully cross-platform.
- Confirm no hidden Windows-only assumptions.
- Enable continued development while traveling.

**Decisions / Trade-offs**
- Use Unix Makefiles generator (skip Ninja for simplicity).
- Accept Debug build for development; optimize later if needed.

**Next**
- Continue feature development on macOS.

---

## 2026-01-19 — VectorStore Implementation (Contiguous Storage)

**Done**
- Implemented `VectorStore` module with:
  - Fixed dimensionality per store.
  - Contiguous storage using `vector<float>` layout (`data[index * dim + j]`).
  - `id -> index` mapping via `unordered_map`.
  - `index -> id` reverse mapping.
  - Tombstone-based logical deletion (`alive[index]`).
- Added APIs:
  - `insert(id, vector)`
  - `get_ptr(index)`
  - `contains(id)`
  - `remove(id)`
  - `is_alive(index)`
- Added sanity checks in `main.cpp` to verify insert/remove behavior.

**Why**
- Contiguous memory layout improves cache locality for distance computation.
- Internal numeric index provides stable node identifiers for HNSW graph.
- Logical deletion avoids expensive data shifting and graph rewiring.

**Decisions / Trade-offs**
- Insert is append-only; deleted slots are not reused in MVP.
- No compaction or rebuild yet (planned for later).
- Exceptions used for invalid operations to surface bugs early.

**Verification**
- Confirmed:
  - Indices assigned sequentially.
  - Deleted vectors are inaccessible via `get_ptr`.
  - Storage remains contiguous after deletion.

**Next**
- Implement brute-force baseline search (exact topK) for correctness validation.

---

## Current Status Summary

**Completed**
- Project scaffold (CMake, Git, VSCode)
- Cross-platform build (Windows + macOS)
- Distance abstraction (L2 / cosine)
- VectorStore (contiguous storage + tombstone)

**In Progress**
- Documentation backfill (`docs/overview`, `architecture`, `design_decisions`)

**Upcoming**
- Brute-force baseline search (heap-based topK)
- Recall@K evaluation
- HNSW index (incremental: layer 0 → multi-layer)
- Persistence (save/load)
- Benchmarking and performance notes

---

## Guiding Principle Going Forward

Correctness → Measurability → Approximation → Performance

No optimization or concurrency will be introduced
before correctness is validated against brute-force results.


## 2026-01-19 — Bruteforce Baseline (Exact TopK)

**Done**
- Implemented exact topK brute-force search with a size-k max-heap.
- Added demo to compare results and establish a correctness oracle.

**Why**
- Provides ground truth for evaluating HNSW recall@K and debugging.

**Next**
- Add recall@K evaluation harness, then start HNSW layer-0 index.

## 2026-01-24 — Recall@K & Benchmark Harness

**Done**
- Added an evaluation harness to compute recall@K and average query latency.
- Designed harness to accept a generic search function so HNSW can be plugged in later.

**Why**
- Enables measurable validation of approximate search quality against brute-force ground truth.

**Next**
- Implement HNSW (start with layer-0) and evaluate recall/latency trade-offs.

## 2026-01-24 — HNSW Layer-0 Index

**Done**
- Implemented HNSW layer-0 graph index with:
  - search-driven insertion (efConstruction)
  - best-first layer search (efSearch)
  - degree constraint via pruning (M)
- Integrated HNSW layer-0 into recall@K evaluation harness.

**Why**
- Establishes a measurable approximate search baseline before implementing hierarchy and heuristics.

**Next**
- Add hierarchical levels (random_level + greedy descent) and neighbor diversity heuristic.

## 2026-01-24 — ef_search Sweep Evaluation

**Done**
- Added ef_search sweep to evaluation harness.
- Measured recall@K and latency trade-offs for HNSW layer-0.

**Why**
- Quantifies quality–performance trade-off in approximate search.
- Establishes baseline metrics for future index improvements.

**Next**
- Add neighbor diversity heuristic.
- Implement hierarchical HNSW.


## 2026-01-24 — Neighbor Diversity Heuristic (HNSW0)

**Done**
- Implemented HNSW neighbor diversity heuristic for:
  - insertion neighbor selection
  - degree pruning
- Added a fallback fill step to ensure up to M neighbors are kept.

**Why**
- Prevents neighbor clustering, improves graph navigability.
- Improves recall at the same ef_search, especially at large N.

**Next**
- Re-run ef_search sweep and compare curves (diversity on/off).
- Implement hierarchical HNSW (multi-level) for faster navigation.

## 2026-01-24 — Hierarchical HNSW (Multi-level)

**Done**
- Implemented hierarchical HNSW with:
  - random level assignment
  - greedy descent on upper layers (ef=1)
  - efConstruction search + connect + prune per layer
  - layer-0 search with efSearch
- Kept neighbor diversity heuristic as a configurable option.
- Used per-node adjacency storage to avoid huge per-layer vector overhead at large N.

**Why**
- Layer-0-only ANN degrades at large N.
- Hierarchical navigation enables coarse-to-fine routing, improving recall/latency trade-off.

**Next**
- Run A/B (diversity on/off) sweeps on hierarchical HNSW.
- Compare against HNSW0 curves to quantify improvements.

## 2026-01-24 — Search Optimization: visited stamp

**Done**
- Replaced `unordered_set` visited tracking with a cache-friendly stamp-array (`Visited`).
- Reused a single visited buffer inside Hnsw0/Hnsw to avoid per-search allocations.
- Added overflow-safe stamp reset.

**Why**
- `unordered_set` incurs heavy hashing and cache-miss overhead at large N / high ef_search.
- Stamp-array makes visited checks O(1) contiguous memory access.

**Expected impact**
- Lower query latency (especially for large ef_search) with minimal/no recall change.


---

## Milestone: Hierarchical HNSW + Diversity + Persistence (MVP Closed Loop)

**Context**

This milestone completes the first fully runnable VecDB MVP:
from raw vectors → ANN index → evaluation → persistence → reload & query.

All components now form a closed loop and can be executed end-to-end on both
Windows and macOS.

---

### What was implemented

#### 1. Hierarchical HNSW (Full)

- Implemented multi-layer HNSW with random level assignment
- Upper layers are sparse, lower layers dense
- Search uses greedy descent from top layer, then ef_search expansion at layer 0
- Parameters:
  - M / M0
  - ef_construction
  - ef_search

#### 2. Neighbor Diversity Heuristic

- Added optional diversity heuristic during neighbor selection
- Prevents neighbors from collapsing into a single local region
- A/B comparison:
  - Index A: diversity OFF
  - Index B: diversity ON
- Empirical results show higher recall@k for diversity-enabled index
  with comparable or better latency

#### 3. Evaluation Harness (Recall / Latency Curve)

- Ground truth computed via brute-force TopK
- ANN results compared against ground truth
- Metrics:
  - recall@k
  - average latency per query
- ef_search sweep (10 → 200) shows expected quality–performance tradeoff

Example (N=200k, dim=32, k=10):

- ef_search=50:
  - recall ≈ 0.78–0.82
  - latency ≈ 0.1 ms
- ef_search=200:
  - recall ≈ 0.96–0.99
  - latency ≈ 0.3 ms

#### 4. Persistence (Save / Open)

- Added disk persistence for both VectorStore and HNSW graph
- Saved artifacts:
  - manifest.json
  - vectors.bin
  - alive.bin
  - ids.txt
  - hnsw.bin
- Tombstone deletion preserves index stability across restarts
- `Collection::open()` restores index without rebuild

Persistence demo verified:
- save → exit → open → search
- Returned nearest neighbors and distances match expected L2^2 values

---

### Design choices

- Chose correctness-first:
  - upsert/remove invalidates index
  - index rebuilt explicitly
- Index stability prioritized over compaction
- Binary format used for graph persistence for efficiency

---

### Known limitations (MVP)

- No concurrent reads/writes
- No incremental index update
- No crash-safe atomic writes (WAL/checkpoint not implemented)
- No SIMD / mmap / compression optimizations

---

### Status

- Core ANN functionality complete
- Persistence verified
- Ready to proceed with:
  - unit / integration tests
  - README & delivery documentation

---

## Milestone: Tests (Unit + Integration)

Added a standalone test executable `vecdb_tests` (no external frameworks) and a CTest target.

### What is covered
- Distance sanity checks (L2^2, cosine, normalization)
- VectorStore operations (upsert/update, tombstone deletion, stable indices)
- Bruteforce correctness on a manual toy dataset
- HNSW recall sanity on a small random dataset
- Persistence roundtrip: create → upsert → build_index → save → open → search

### How to run
```bash
cmake -S . -B build
cmake --build build -j
./build/vecdb_tests
# or:
cd build && ctest --output-on-failure
