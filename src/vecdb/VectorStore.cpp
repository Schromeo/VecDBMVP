#include "VectorStore.h"

#include <algorithm>
#include <stdexcept>

namespace vecdb {

VectorStore::VectorStore(std::size_t dim) : dim_(dim) {
  if (dim_ == 0) throw std::invalid_argument("VectorStore: dim must be > 0");
}

void VectorStore::validate_dim_(const std::vector<float>& vec) const {
  if (vec.size() != dim_) {
    throw std::invalid_argument("VectorStore: vector dim mismatch");
  }
}

float* VectorStore::ptr_at_(std::size_t index) {
  return data_.data() + index * dim_;
}

const float* VectorStore::ptr_at_(std::size_t index) const {
  return data_.data() + index * dim_;
}

bool VectorStore::is_alive(std::size_t index) const {
  if (index >= alive_.size()) return false;
  return alive_[index] != 0;
}

bool VectorStore::contains(const std::string& id) const {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) return false;
  std::size_t idx = it->second;
  return is_alive(idx);
}

const std::string& VectorStore::id_at(std::size_t index) const {
  if (index >= ids_.size()) {
    throw std::out_of_range("VectorStore::id_at: index out of range");
  }
  return ids_[index];
}

const float* VectorStore::get_ptr(std::size_t index) const {
  if (index >= size()) return nullptr;
  if (!is_alive(index)) return nullptr;
  return ptr_at_(index);
}

float* VectorStore::get_mut_ptr(std::size_t index) {
  if (index >= size()) return nullptr;
  if (!is_alive(index)) return nullptr;
  return ptr_at_(index);
}

const float* VectorStore::get_ptr(const std::string& id) const {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) return nullptr;
  std::size_t idx = it->second;
  return get_ptr(idx);
}

float* VectorStore::get_mut_ptr(const std::string& id) {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) return nullptr;
  std::size_t idx = it->second;
  return get_mut_ptr(idx);
}

bool VectorStore::try_get_index(const std::string& id, std::size_t& out_index) const {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) return false;
  std::size_t idx = it->second;
  if (!is_alive(idx)) return false;
  out_index = idx;
  return true;
}

std::size_t VectorStore::insert(const std::string& id, const std::vector<float>& vec) {
  validate_dim_(vec);
  if (id.empty()) throw std::invalid_argument("VectorStore::insert: id cannot be empty");

  auto it = id_to_index_.find(id);
  if (it != id_to_index_.end()) {
    std::size_t idx = it->second;
    if (is_alive(idx)) {
      throw std::runtime_error("VectorStore::insert: id already exists");
    }
    // existed but dead -> revive at same index
    std::copy(vec.begin(), vec.end(), ptr_at_(idx));
    alive_[idx] = 1;
    // keep ids_[idx] as id
    return idx;
  }

  // Append new slot
  std::size_t idx = ids_.size();
  ids_.push_back(id);
  alive_.push_back(1);

  data_.resize(ids_.size() * dim_);
  std::copy(vec.begin(), vec.end(), ptr_at_(idx));

  id_to_index_[id] = idx;
  return idx;
}

std::size_t VectorStore::upsert(const std::string& id, const std::vector<float>& vec) {
  validate_dim_(vec);
  if (id.empty()) throw std::invalid_argument("VectorStore::upsert: id cannot be empty");

  auto it = id_to_index_.find(id);
  if (it != id_to_index_.end()) {
    std::size_t idx = it->second;
    // overwrite (even if dead -> revive)
    std::copy(vec.begin(), vec.end(), ptr_at_(idx));
    alive_[idx] = 1;
    if (ids_[idx].empty()) ids_[idx] = id;
    return idx;
  }

  // new id -> append
  std::size_t idx = ids_.size();
  ids_.push_back(id);
  alive_.push_back(1);

  data_.resize(ids_.size() * dim_);
  std::copy(vec.begin(), vec.end(), ptr_at_(idx));

  id_to_index_[id] = idx;
  return idx;
}

bool VectorStore::remove(const std::string& id) {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) return false;

  std::size_t idx = it->second;
  if (!is_alive(idx)) return false;

  alive_[idx] = 0;

  // IMPORTANT:
  // We intentionally keep ids_[idx] and id_to_index_[id] so that an "upsert"
  // can revive the same id at the same stable index during the same run.
  // (Persistence behavior depends on whether ids for dead slots are saved.)
  return true;
}

void VectorStore::clear() {
  data_.clear();
  alive_.clear();
  ids_.clear();
  id_to_index_.clear();
}

void VectorStore::load_from_disk(std::size_t N,
                                 const std::vector<float>& vectors,
                                 const std::vector<std::uint8_t>& alive,
                                 const std::vector<std::string>& ids) {
  if (N == 0) {
    clear();
    return;
  }
  if (alive.size() != N) {
    throw std::runtime_error("VectorStore::load_from_disk: alive size mismatch");
  }
  if (ids.size() != N) {
    throw std::runtime_error("VectorStore::load_from_disk: ids size mismatch");
  }
  if (vectors.size() != N * dim_) {
    throw std::runtime_error("VectorStore::load_from_disk: vectors size mismatch");
  }

  ids_ = ids;
  alive_ = alive;
  data_ = vectors;

  id_to_index_.clear();
  id_to_index_.reserve(N);

  // Rebuild mapping:
  // - If ids_[i] is non-empty, we map it to i (even if dead) so we can "revive".
  // - If empty, skip (hole with no name).
  for (std::size_t i = 0; i < N; ++i) {
    if (!ids_[i].empty()) {
      id_to_index_[ids_[i]] = i;
    }
  }
}

}  // namespace vecdb
