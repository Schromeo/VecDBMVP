#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vecdb {

// VectorStore: contiguous in-memory storage for fixed-dimension vectors.
// - Index is stable (0..N-1). Deletion creates "dead" slots but keeps indices.
// - We maintain id <-> index mapping.
// - This design is critical for persistence + HNSW graph correctness, because
//   HNSW neighbor lists store indices.
class VectorStore {
 public:
  explicit VectorStore(std::size_t dim);

  // Fixed vector dimension for this store.
  std::size_t dim() const { return dim_; }

  // Number of slots (including dead slots). Indices range: [0, size()).
  std::size_t size() const { return ids_.size(); }

  // True if index exists and is alive.
  bool is_alive(std::size_t index) const;

  // Check whether an id exists AND is alive.
  bool contains(const std::string& id) const;

  // Return id string stored at an index (may be empty for dead slots).
  // Precondition: index < size()
  const std::string& id_at(std::size_t index) const;

  // Get pointer to vector data by index.
  // Returns nullptr if index out of range OR slot is dead.
  const float* get_ptr(std::size_t index) const;
  float* get_mut_ptr(std::size_t index);

  // Get pointer to vector data by id (alive only).
  // Returns nullptr if id not found or dead.
  const float* get_ptr(const std::string& id) const;
  float* get_mut_ptr(const std::string& id);

  // Insert a new (id, vec).
  // - If id already exists and alive: throws.
  // - If id exists but is dead: revives at same index (treated like upsert).
  // Returns the index used.
  std::size_t insert(const std::string& id, const std::vector<float>& vec);

  // Upsert (insert or overwrite):
  // - If id exists and alive: overwrite vector in-place, return its index.
  // - If id exists but dead: revive at same index, overwrite vector, return index.
  // - Else: append a new slot, return new index.
  std::size_t upsert(const std::string& id, const std::vector<float>& vec);

  // Remove by id:
  // - If id not found or already dead: returns false.
  // - Else: mark dead, keep data/ids for stable indexing, return true.
  bool remove(const std::string& id);

  // Optional helper: get index for an alive id.
  // Returns true + sets out_index if exists and alive; otherwise false.
  bool try_get_index(const std::string& id, std::size_t& out_index) const;

  // Clear all data.
  void clear();

  // -------- Persistence support --------
  //
  // Rebuild the store exactly as it existed on disk.
  // This MUST preserve indices:
  // - ids[i] is the id for slot i (may be empty if dead)
  // - alive[i] is 1/0 per slot
  // - vectors is a flat array of length N*dim in row-major order
  //
  // After this, id->index mapping is rebuilt for alive slots.
  void load_from_disk(std::size_t N,
                      const std::vector<float>& vectors,
                      const std::vector<std::uint8_t>& alive,
                      const std::vector<std::string>& ids);

 private:
  void validate_dim_(const std::vector<float>& vec) const;

  float* ptr_at_(std::size_t index);
  const float* ptr_at_(std::size_t index) const;

  std::size_t dim_ = 0;

  // Flat array: [v0_dim floats][v1_dim floats]...
  std::vector<float> data_;

  // Slot status (1 = alive, 0 = dead).
  std::vector<std::uint8_t> alive_;

  // Index -> id (empty string allowed for dead slot).
  std::vector<std::string> ids_;

  // id -> index (only for alive ids)
  std::unordered_map<std::string, std::size_t> id_to_index_;
};

}  // namespace vecdb
