#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "Distance.h"
#include "Metadata.h"
#include "VectorStore.h"
#include "Hnsw.h"

namespace vecdb {

// Serializer is responsible for persistence (disk I/O).
//
// v1 on-disk layout (minimal, readable, versioned):
//   <dir>/manifest.json   -- metadata (dim, metric, version, hnsw params)
//   <dir>/vectors.bin     -- contiguous float32 vectors
//   <dir>/alive.bin       -- alive bitmap (uint8_t per index)
//   <dir>/ids.txt         -- index -> id (one per line, empty for dead slots)
//   <dir>/meta.txt        -- index -> metadata (one line per index, key=value;...)
//   <dir>/hnsw.bin        -- HNSW graph structure (binary)
//
// Notes:
// - We keep formats simple and explicit for clarity.
// - Backward compatibility is managed via manifest.version.
//
class Serializer {
 public:
  // -------- Manifest --------
  struct Manifest {
    int version = 1;
    std::size_t dim = 0;
    Metric metric = Metric::L2;
    Hnsw::Params hnsw_params{};
  };

  // Read / write manifest.json
  static Manifest read_manifest(const std::string& dir);
  static void write_manifest(const std::string& dir, const Manifest& mf);

  // -------- VectorStore --------

  // Save all vector data, alive flags, and id mapping.
  static void save_store(const std::string& dir, const VectorStore& store);

  // Load vector data, alive flags, and id mapping into an existing store.
  // The store is expected to be constructed with the correct dim.
  static void load_store(const std::string& dir, VectorStore& store);

  // -------- HNSW --------

  // Save HNSW graph structure to disk.
  // Requires: index already built, store provided for size checks.
  static void save_hnsw(const std::string& dir,
                        const Hnsw& hnsw,
                        const VectorStore& store);

  // Load HNSW graph structure from disk into an existing Hnsw object.
  // Requires: store already loaded and index object constructed.
  static void load_hnsw(const std::string& dir,
                        Hnsw& hnsw,
                        const VectorStore& store);
};

}  // namespace vecdb
