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
