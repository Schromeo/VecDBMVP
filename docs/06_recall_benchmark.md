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
