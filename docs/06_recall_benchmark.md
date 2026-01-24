# Recall@K & Benchmark Harness

## Purpose
Provide a measurable evaluation framework for approximate nearest neighbor search.

Bruteforce search is used as ground truth. Approximate search (HNSW) is evaluated by:
- recall@K
- average query latency

## Definitions

### recall@K
Let:
- `GT` be the set of topK indices from brute-force
- `AP` be the set of topK indices from approximate search

Then:
recall@K = |GT ∩ AP| / |GT|

Values:
- 1.0 = perfect match to brute-force topK
- lower = missing more true neighbors

## Harness Design
The evaluator accepts two search functions:
- `truth(query, k)` -> results
- `approx(query, k)` -> results

This decouples evaluation from index implementation and allows plugging in HNSW later.

## Complexity
For `Q` queries:
- truth cost: Q * O(N*D)
- approx cost: Q * (depends on index)
- recall computation: O(K) per query using a hash set
