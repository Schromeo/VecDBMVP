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
- Metadata:
  - per-vector key/value map
  - exact-match filtering
- Concurrency:
  - multi-reader/single-writer locking in `Collection`

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

## CLI + CSV

The executable supports a simple CLI for CSV ingest and search.

### Quick start (CLI)

```bash
cmake -S . -B build
cmake --build build -j
python scripts/generate_csv.py
./build/vecdb.exe create --dir data/my_collection --dim 16 --metric l2
./build/vecdb.exe load --dir data/my_collection --csv data/vectors.csv --header
./build/vecdb.exe build --dir data/my_collection --M 16 --M0 32 --efC 100 --diversity 1
./build/vecdb.exe search --dir data/my_collection --query_csv data/queries.csv --k 5 --ef 50 --header
````

### CSV format

- **vectors.csv**: `id,v0,v1,...,v(dim-1)`
- **queries.csv**: `v0,v1,...,v(dim-1)` (or `id,v0,...` with `--has-id`)

Notes:
- Use `--header` if your CSV has a header row.
- Use `--has-id` if your id column is numeric (e.g., `123,0.1,0.2,...`).
- Use `--meta` if your CSV has a trailing metadata column (`key=value;key2=value2`).

### CLI parameter summary

| Command | Required | Optional |
| --- | --- | --- |
| `create` | `--dir`, `--dim` | `--metric`, `--M`, `--M0`, `--efC`, `--diversity`, `--seed`, `--level_mult` |
| `load` | `--dir`, `--csv` | `--header`, `--meta`, `--build` |
| `build` | `--dir` | `--metric`, `--M`, `--M0`, `--efC`, `--diversity`, `--seed`, `--level_mult` |
| `search` | `--dir`, (`--query` or `--query_csv`) | `--k`, `--ef`, `--limit`, `--header`, `--has-id`, `--filter` |
| `stats` | `--dir` | - |

### Create → load → build

```bash
./build/vecdb.exe create --dir data/my_collection --dim 128 --metric l2
./build/vecdb.exe load --dir data/my_collection --csv data/vectors.csv --header
./build/vecdb.exe build --dir data/my_collection --M 16 --M0 32 --efC 100 --diversity 1
````

### Search with CSV queries

```bash
./build/vecdb.exe search --dir data/my_collection --query_csv data/queries.csv --k 10 --ef 100 --header
````

### Load with metadata + filter

```bash
./build/vecdb.exe load --dir data/my_collection --csv data/vectors_with_meta.csv --header --meta
./build/vecdb.exe search --dir data/my_collection --query_csv data/queries.csv --k 10 --ef 100 --header --filter cluster=2
````

### Generate sample CSVs

```bash
python scripts/generate_csv.py
````

This script produces:
- `data/vectors.csv` (string id)
- `data/vectors_numeric_id.csv` (numeric id → use `--has-id`)
- `data/vectors_with_meta.csv` (metadata column → use `--meta`)
- `data/queries.csv` (no id)
- `data/queries_with_id.csv` (string id)

---

## Documentation

Detailed design notes and implementation walkthroughs are available in `docs/`:

* Architecture and design decisions
* Vector storage
* HNSW (layer-0 and full hierarchical)
* Evaluation methodology
* Development log

---

## API Reference (C++)

Core API lives in `vecdb::Collection`.

### Create / Load / Save

```cpp
#include "vecdb/Collection.h"

vecdb::Collection::Options opt;
opt.dim = 128;
opt.metric = vecdb::Metric::L2;
opt.hnsw_params.M = 16;
opt.hnsw_params.M0 = 32;
opt.hnsw_params.ef_construction = 100;
opt.hnsw_params.use_diversity = true;

auto col = vecdb::Collection::create("data/my_collection", opt);
col.upsert("id_1", std::vector<float>(128, 0.1f));
col.build_index();
col.save();

auto col2 = vecdb::Collection::open("data/my_collection");
````

### Search

```cpp
std::vector<float> q(128, 0.1f);
auto res = col2.search(q, /*k=*/10, /*ef_search=*/100);
for (auto& r : res) {
  std::cout << r.index << " " << col2.id_at(r.index) << " " << r.distance << "\n";
}
````

### Metadata + Filter

```cpp
vecdb::Metadata meta;
meta["cluster"] = "2";
meta["source"] = "synthetic";
col.upsert("id_1", std::vector<float>(128, 0.1f), meta);

vecdb::Collection::MetadataFilter f;
f.key = "cluster";
f.value = "2";
auto res2 = col.search(q, /*k=*/10, /*ef_search=*/100, f);
````

---

## Design Doc (Summary)

- **Storage**: `VectorStore` keeps contiguous vectors with stable indices and tombstone deletion.
- **Index**: Hierarchical HNSW with configurable `M`, `M0`, `ef_construction`, `ef_search`, and optional diversity heuristic.
- **Persistence**: `Serializer` writes a manifest, vector store, and HNSW graph. Collections can be re-opened without rebuilding.

---

## Trade-offs & Future Improvements

- **Index rebuilds**: current design invalidates HNSW on any mutation for correctness; incremental updates are a planned improvement.
- **CSV parsing**: supports headers and quoted fields, but not full RFC edge cases (e.g., newlines inside quoted fields).
- **Metadata filtering**: currently uses exact scan for correctness (no ANN acceleration).
- **Concurrency**: multi-reader/single-writer locks; no transactions.
- **Metadata**: simple key=value map only; no schema, no typed values, no index.
- **Durability**: persistence is simple binary files; no WAL or atomic checkpointing.

---

## Simple Benchmark

The built-in demo includes an HNSW recall/latency sweep.

Example (N=200k, dim=32, k=10):

- ef_search=50: recall ≈ 0.78–0.82, avg latency ≈ 0.1 ms
- ef_search=200: recall ≈ 0.96–0.99, avg latency ≈ 0.3 ms

Numbers are approximate and depend on hardware.

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

Verbose output:

```bash
tests/run_all_tests_verbose.ps1
# or: ctest -V --test-dir build
````

### Coverage

* Distance: L2^2 / cosine / normalize sanity
* VectorStore: upsert/update, tombstone deletion, stable indices
* Bruteforce: correctness on a small manual dataset
* HNSW: average recall sanity on small random dataset
* Persistence: create → upsert → build_index → save → open → search (top1 + distance)


