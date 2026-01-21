# VectorStore Module

This document describes the VectorStore, which is the data plane of the VecDB MVP.
It is responsible for storing vector embeddings and providing stable access
to them for search algorithms.

---

## Responsibilities

VectorStore is responsible for:
- Storing vector embeddings in memory
- Enforcing fixed dimensionality
- Mapping external IDs to internal indices
- Supporting logical deletion
- Providing fast, cache-friendly access for distance computation

VectorStore does **not**:
- Perform similarity search
- Handle concurrency
- Manage persistence (handled by a separate module)

---

## Data Model

### External ID vs Internal Index

- **External ID** (`string`)
  - User-facing identifier
  - Arbitrary, not necessarily numeric

- **Internal Index** (`size_t`)
  - Dense, zero-based
  - Used internally by search algorithms and graph indices

This separation allows:
- Contiguous memory layout
- Compact graph representations
- Stable references even when vectors are deleted

---

## Memory Layout

Vectors are stored in a single contiguous array:
```
data_ = [
v0[0], v0[1], ..., v0[d-1],
v1[0], v1[1], ..., v1[d-1],
...
]

```

Access pattern:
```

vector i starts at &data_[i * dim]

```

### Benefits
- Cache-friendly sequential access
- Efficient distance computation
- Simple serialization for persistence

---

## Deletion Strategy (Tombstone)

VectorStore uses **logical deletion**:

- `alive[index] == 1` → vector is active
- `alive[index] == 0` → vector is deleted

Deleted vectors:
- Remain in memory
- Are skipped during search
- Preserve index stability

### Why tombstones?
- Hard deletion would require shifting data and updating all graph edges
- Logical deletion keeps indices stable
- Space can be reclaimed later via rebuild/compaction

---

## API Overview

### `insert(id, vector) -> index`

**Purpose**
- Insert a new vector and assign a stable internal index.

**Preconditions**
- `vector.size() == dim`
- `id` not already alive

**Postconditions**
- Vector appended to contiguous storage
- New index assigned and marked alive

**Complexity**
- Time: O(dim)
- Space: O(dim)

---

### `get_ptr(index) -> const float*`

**Purpose**
- Provide raw pointer access for distance computation.

**Behavior**
- Returns `nullptr` if index is invalid or deleted
- Otherwise returns pointer to contiguous data

---

### `contains(id) -> bool`

Checks whether an ID exists and is alive.

---

### `remove(id) -> bool`

**Purpose**
- Logically delete a vector by ID.

**Behavior**
- Marks vector as deleted
- Does not reclaim memory

---

## Design Trade-offs

### Append-only insertion
- Simplifies index assignment
- Enables trivial persistence
- Avoids fragmentation

### No index reuse in MVP
- Simpler correctness model
- Reuse can be added during rebuild

### No concurrency
- Avoids race conditions during early development
- Future extension: multi-reader/single-writer locking

---

## Role in the System

VectorStore is the foundation for:
- Bruteforce exact search
- HNSW approximate index
- Persistence layer (save/load)

All higher-level algorithms operate on **internal indices**
provided by VectorStore.

