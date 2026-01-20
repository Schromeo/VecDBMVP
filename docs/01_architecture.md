# Architecture

## Components

### VectorStore (data plane)
- Stores embeddings in a contiguous `vector<float>` layout.
- Maintains `id -> index` and `index -> id`.
- Uses tombstone (`alive[index]`) for logical deletion.

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
