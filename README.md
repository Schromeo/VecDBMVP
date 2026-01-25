# VecDB MVP

A minimal, from-scratch vector database prototype implemented in C++.

This project explores the core building blocks of modern vector databases:
vector storage, approximate nearest neighbor search (HNSW), evaluation, and
persistence.

---

## Features

- Vector distance metrics:
  - L2 squared distance
  - Cosine distance
- Contiguous vector storage with stable indices
- Tombstone deletion (index stability)
- Approximate nearest neighbor search:
  - Hierarchical HNSW
  - Configurable M / M0 / ef_construction / ef_search
  - Optional neighbor diversity heuristic
- Evaluation harness:
  - brute-force ground truth
  - recall@k and latency measurement
- Persistence:
  - save and reload vector store and HNSW index
  - no rebuild required after restart

---

## Build

### Windows

```bash
cmake -S . -B build
cmake --build build -j
./build/vecdb.exe
````

### macOS

```bash
cmake -S . -B build
cmake --build build -j
./build/vecdb
```

---

## Demo Output

The executable runs:

1. Distance sanity checks
2. VectorStore sanity checks
3. Brute-force TopK demo
4. HNSW recall/latency benchmark
5. Persistence demo (save → open → search)

---

## Documentation

Detailed design notes and implementation walkthroughs are available in `docs/`:

* Architecture and design decisions
* Vector storage
* HNSW (layer-0 and full hierarchical)
* Evaluation methodology
* Development log

---

## Status

This project is an MVP intended for learning and exploration.

Planned next steps:

* unit and integration tests
* incremental index updates
* concurrency support
* more robust persistence (WAL, atomic writes)

## Tests

This repo includes a minimal unit + integration test runner (no external deps).

### Build & run tests

```bash
cmake -S . -B build
cmake --build build -j
./build/vecdb_tests   # Windows: vecdb_tests.exe
````

Or via CTest:

```bash
cd build
ctest --output-on-failure
```

### Coverage

* Distance: L2^2 / cosine / normalize sanity
* VectorStore: upsert/update, tombstone deletion, stable indices
* Bruteforce: correctness on a small manual dataset
* HNSW: average recall sanity on small random dataset
* Persistence: create → upsert → build_index → save → open → search (top1 + distance)


