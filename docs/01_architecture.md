# Architecture

## Components

### VectorStore (data plane)
- Stores embeddings in a contiguous `vector<float>` layout.
- Maintains `id -> index` and `index -> id`.
- Uses tombstone (`alive[index]`) for logical deletion.
- Stores optional metadata per vector (string key/value map).

### Distance (metric abstraction)
- Unified `distance(metric, a, b)` API.
- L2 uses squared distance for faster comparisons.
- Cosine uses `1 - cosine_similarity`; optionally normalize at insert-time.

### Index (HNSW, control plane)
- Multi-layer adjacency lists: `layers[level][node] -> neighbors`.
- Parameters: M, efConstruction, efSearch.
- Insert is search-driven: find candidates then connect + prune.

### Baseline (BruteForce)
- O(N*D) exact topK for correctness validation and recall@K.

### Filtering (Metadata)
- Metadata filtering is implemented as an exact scan that checks key/value pairs
	before computing distances.
- This keeps correctness simple; ANN-aware filtering is a future improvement.

### Concurrency
- `Collection` uses a shared mutex for multi-reader/single-writer access.
- Reads (search, stats) take shared locks; writes (upsert/build/save/load) take exclusive locks.
