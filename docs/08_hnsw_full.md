# Hierarchical HNSW (Full)

This document describes the multi-level HNSW implementation.

## Why Hierarchy?

Layer-0-only graph search degrades at large N:
- recall drops unless ef_search becomes large
- latency increases because search must explore many nodes

Hierarchical HNSW adds coarse-to-fine navigation:
- upper layers are sparse “highways”
- lower layers are dense “local streets”
This reduces the number of visited nodes needed to reach high recall.

---

## Data Structure

Each node stores adjacency lists only up to its own level.

- Node i has a random `level_i >= 0`.
- For each level `l` in `[0..level_i]`:
  - `links[i][l]` is the neighbor list at that level.

This is memory efficient because most nodes have small levels.

---

## Parameters

- `M`: max degree for upper layers (l >= 1)
- `M0`: max degree for layer 0 (often 2*M)
- `efConstruction`: candidate pool size during insertion
- `efSearch`: candidate pool size during query search
- `use_diversity`: enables neighbor diversity heuristic

---

## Random Level Assignment

Each inserted node receives a random level using a geometric-like distribution.

Intuition:
- many nodes have level 0
- fewer nodes appear in higher layers
- the number of layers grows slowly with N (roughly logarithmic)

---

## Insertion Algorithm (Coarse-to-Fine)

Given a node v with level L:

1. If this is the first node:
   - set it as entry point
   - set max_level = L

2. Otherwise:
   - Start from entry point at current max_level
   - For layers l = max_level down to L+1:
     - greedy descent (ef=1) to move closer to v
   - For layers l = min(L, max_level) down to 0:
     - run search on layer l with efConstruction
     - select neighbors (diverse if enabled)
     - connect bidirectionally on that layer
     - prune degrees to <= M (or M0 on layer 0)

3. If L > max_level:
   - update entry point to v
   - update max_level

---

## Query Algorithm

1. Start from entry point at max_level
2. For l = max_level down to 1:
   - greedy descent on layer l (ef=1)
3. On layer 0:
   - run best-first search with efSearch
   - return topK

---

## Neighbor Diversity Heuristic

When selecting up to M neighbors, iterate candidates in ascending distance to base node.
Accept candidate c only if for all selected neighbors s:

`dist(c, base) < dist(c, s)`

If too few neighbors are selected, fill remaining slots with nearest candidates.

---

## Expected Effect

Compared with layer-0-only:

- At the same ef_search:
  - higher recall (better navigation)
  - often lower latency due to fewer expansions
- At the same recall:
  - noticeably lower latency (coarse-to-fine routing)
