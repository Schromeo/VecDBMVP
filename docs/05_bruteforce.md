# Bruteforce Baseline

## Purpose
Provide an exact topK nearest-neighbor search baseline for correctness validation.

This baseline is used to:
- validate HNSW search results (recall@K)
- debug indexing/search logic
- serve as a correctness oracle during MVP

## API
### `Bruteforce::search(query, k) -> vector<SearchResult>`
- Returns up to `k` nearest alive vectors in ascending distance order.
- Throws if query dimension mismatches the store dimension.

## Algorithm
- Iterate all alive vectors `i in [0..N-1]`
- Compute distance `d(query, vec_i)`
- Maintain a max-heap of size `k`:
  - heap top = current worst among best-k
  - if new distance is smaller, replace worst

Finally, sort heap outputs in ascending distance.

## Complexity
- Time: O(N * D + N log K)
- Space: O(K)

## Notes
- L2 uses squared distance (no sqrt) since it preserves ordering.
- Cosine distance uses `1 - cosine_similarity` via Distance module.
