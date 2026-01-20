#include "VectorStore.h"

#include <stdexcept>

namespace vecdb {

VectorStore::VectorStore(std::size_t dim) : dim_(dim) {
  if (dim_ == 0) {
    throw std::invalid_argument("VectorStore: dim must be > 0");
  }
}

bool VectorStore::contains(const std::string& id) const {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) return false;
  return is_alive(it->second);
}

std::size_t VectorStore::insert(const std::string& id, const std::vector<float>& v) {
  if (v.size() != dim_) {
    throw std::invalid_argument("VectorStore::insert: vector dim mismatch");
  }

  auto it = id_to_index_.find(id);
  if (it != id_to_index_.end() && is_alive(it->second)) {
    throw std::runtime_error("VectorStore::insert: id already exists and alive");
  }

  // If id existed but was deleted, we keep it simple: treat as new record (new index).
  // (Production systems may reuse slots; MVP doesn't.)
  std::size_t index = index_to_id_.size();

  index_to_id_.push_back(id);
  id_to_index_[id] = index;
  alive_.push_back(static_cast<std::uint8_t>(1));

  // Append to contiguous data array
  data_.insert(data_.end(), v.begin(), v.end());

  return index;
}

bool VectorStore::remove(const std::string& id) {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) return false;
  std::size_t idx = it->second;
  if (!is_alive(idx)) return false;
  alive_[idx] = static_cast<std::uint8_t>(0);
  return true;
}

const float* VectorStore::get_ptr(std::size_t index) const {
  if (index >= size()) return nullptr;
  if (!is_alive(index)) return nullptr;
  return &data_[index * dim_];
}

const std::string& VectorStore::id_at(std::size_t index) const {
  if (index >= size()) return empty_id_;
  return index_to_id_[index];
}

bool VectorStore::is_alive(std::size_t index) const {
  if (index >= alive_.size()) return false;
  return alive_[index] != 0;
}

} // namespace vecdb
