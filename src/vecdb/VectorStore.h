#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vecdb {

class VectorStore {
 public:
  explicit VectorStore(std::size_t dim);

  std::size_t dim() const { return dim_; }
  std::size_t size() const { return index_to_id_.size(); }

  bool contains(const std::string& id) const;

  // Insert a new vector. If id already exists and is alive, throws.
  // Returns internal index [0..N-1].
  std::size_t insert(const std::string& id, const std::vector<float>& v);

  // Logical delete by id. Returns true if deleted, false if not found/already dead.
  bool remove(const std::string& id);

  // Access vector by internal index. Returns nullptr if index invalid or dead.
  const float* get_ptr(std::size_t index) const;

  // Convenience: get id by index (empty string if invalid)
  const std::string& id_at(std::size_t index) const;

  // Whether an internal index is alive
  bool is_alive(std::size_t index) const;

 private:
  std::size_t dim_;
  std::vector<float> data_;                  // contiguous: N * dim_
  std::unordered_map<std::string, std::size_t> id_to_index_;
  std::vector<std::string> index_to_id_;     // index -> id
  std::vector<std::uint8_t> alive_;          // 1 = alive, 0 = deleted
  std::string empty_id_;
};

} // namespace vecdb
