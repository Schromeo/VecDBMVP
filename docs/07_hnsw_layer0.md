# HNSW Layer-0 (MVP)

This document describes the MVP HNSW implementation limited to layer-0 only.
It provides an approximate nearest neighbor index using a graph-based search.

Layer-0 is implemented first to:
- validate core graph search logic
- enable measurable recall/latency trade-offs
- keep the system understandable before adding hierarchy

---

## Data Structure

- The index maintains an undirected graph:
  - `neighbors_[i]` is the adjacency list of node `i`
- Nodes are internal indices from VectorStore (`size_t`).

### Degree Constraint
- Each node has at most `M` neighbors.
- After insertion and edge updates, neighbor lists are pruned to enforce the constraint.

---

## Parameters

### `M`
- Maximum degree per node.
- Larger M:
  - more edges
  - better connectivity (often better recall)
  - slower insert and higher memory usage

### `efConstruction`
- Candidate pool size during insertion.
- Larger efConstruction:
  - higher-quality neighbor selection
  - slower insertion

### `efSearch`
- Candidate pool size during search.
- Larger efSearch:
  - higher recall
  - slower query latency

---

## Search Algorithm (Layer-0)

Given:
- query vector `q`
- entry point `ep`
- efSearch

Maintain:
- `candidates`: min-heap ordered by distance to `q` (best-first expansion)
- `results`: max-heap storing current best `efSearch` nodes (worst on top)
- `visited`: to avoid revisiting nodes

Stop condition:
- If the best candidate in `candidates` is worse than the worst node in `results`,
  further exploration cannot improve the result set.

Output:
- Sort the `results` by distance ascending
- Return topK

---

## Insertion Algorithm (Search-driven)

To insert node `v`:
1. Use `v` as a query vector and run layer-0 search with `efConstruction` to get candidate neighbors.
2. Select up to `M` nearest candidates (simple MVP rule).
3. Connect bidirectionally: `v <-> neighbor`.
4. Prune neighbor lists to maintain degree ≤ M.

Notes:
- MVP uses a simple selection rule (nearest M). Later versions add diversity heuristics.

---

## Complexity (High-level)

Search:
- depends on graph structure and efSearch
- roughly O(efSearch * log efSearch + visited * D)

Insert:
- dominated by efConstruction search + pruning
- roughly O(efConstruction * log efConstruction + M log M + distance computations)

---

## Role in the System

- Provides the first approximate index plugged into the evaluation harness.
- Enables recall@K and latency trade-off measurement before adding hierarchy.


## Neighbor Diversity Heuristic (7B)

MVP initially selected the nearest M neighbors directly, which can cause
neighbors to cluster in the same local region and reduce navigability.

We add the classic HNSW diversity heuristic:

Iterate candidates in ascending distance to the base node.
Accept candidate `c` only if for every already selected neighbor `s`:
`dist(c, base) < dist(c, s)`.

Intuition:
- Reject candidates that are too close to an already selected neighbor.
- Keep neighbors that provide diverse directions, improving graph navigability.

If the heuristic is too strict and selects fewer than M neighbors, we fill
remaining slots with nearest candidates to ensure connectivity.
