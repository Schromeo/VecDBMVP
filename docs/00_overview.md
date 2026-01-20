# VecDB MVP Overview

## Goal
Build a minimal viable vector database supporting insert + similarity search + persistence (future).

## MVP Scope
- VectorStore: contiguous float storage + id->index mapping + tombstone
- Distance: L2^2 and cosine distance (with optional normalization)
- Search: brute-force baseline + HNSW (incremental)

## Non-Goals (for MVP)
- High concurrency / transactions
- Updates / hard deletes (use tombstone)
- Advanced filtering / metadata indexing
- Distributed deployment

## How to Build & Run
(留空，之后 README 写也行)
