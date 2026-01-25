# Recall Benchmark and ef_search Sweep

This document evaluates the recall–latency trade-off of the HNSW layer-0 index.

## Setup
- Dataset size: N = 5000
- Dimension: 32
- Number of queries: 200
- k = 10
- Metric: L2
- Index: HNSW layer-0
- Parameters:
  - M = 16
  - efConstruction = 100

## ef_search Sweep Results

| ef_search | recall@10 | avg_latency_ms |
|-----------|-----------|----------------|
| 10        |           |                |
| 20        |           |                |
| 50        |           |                |
| 100       |           |                |
| 200       |           |                |

## Observations
- Increasing ef_search consistently improves recall.
- Latency grows roughly linearly with ef_search.
- Moderate ef_search values already achieve strong recall.
- These results establish a baseline for future improvements
  (neighbor diversity heuristic and hierarchical HNSW).

## Next Steps
- Improve recall under the same ef_search using neighbor diversity heuristics.
- Reduce latency at the same recall using hierarchical HNSW.

### A/B: Neighbor Diversity Heuristic

We compare HNSW layer-0 with neighbor diversity heuristic ON vs OFF under the same setup:
N=500000, dim=32, queries=200, k=10, M=16, efConstruction=100.

Result: diversity improves recall significantly for medium/high ef_search.
For example:
- ef_search=50: recall@10 improves from 0.4645 -> 0.5070
- ef_search=100: recall@10 improves from 0.6125 -> 0.6730
- ef_search=200: recall@10 improves from 0.7680 -> 0.8150

Latency increases moderately, while staying in the same order of magnitude.
This indicates improved graph navigability from diversified neighbor selection.

Next step: implement hierarchical HNSW to reduce latency for the same recall by enabling coarse-to-fine navigation.
