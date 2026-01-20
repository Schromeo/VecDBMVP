# Design Decisions & Trade-offs

## 1) id vs index separation
Decision:
- Use stable internal `index` for storage and graph nodes; map external `id` to `index`.
Why:
- Enables contiguous storage and simple persistence; graph stores integers only.

## 2) Contiguous vector storage
Decision:
- Store embeddings in `vector<float> data` with layout `data[index * dim + j]`.
Why:
- Cache-friendly distance computation; single bulk write/read for persistence.

## 3) Deletion strategy
Decision:
- Use tombstone (logical delete) in MVP; no immediate graph repair.
Why:
- Hard delete breaks index stability and requires expensive graph rewiring; rebuild can reclaim space later.

## 4) Metric implementation
Decision:
- Use squared L2 distance; cosine distance = 1 - cosine similarity.
Why:
- Squared L2 preserves ordering without sqrt; cosine unified into “smaller is closer”.

## 5) Why no high concurrency in MVP
Decision:
- Single-threaded correctness first; future: RWLock (multi-reader/single-writer).
Why:
- Concurrency hides correctness bugs and complicates persistence guarantees.
