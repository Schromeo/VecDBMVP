#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace vecdb {

// Visited set implemented as "stamp array".
// mark[i] == stamp  => visited in current search
// This avoids unordered_set and is cache-friendly.
//
// NOTE: Not thread-safe. (MVP: single-thread search)
class Visited {
 public:
  Visited() = default;

  // Start a new search context for universe size n.
  // Ensures internal array size >= n, then increments stamp.
  void start(std::size_t n) {
    ensure_size(n);
    advance_stamp();
  }

  bool test(std::size_t i) const {
    return (i < mark_.size()) && (mark_[i] == stamp_);
  }

  void set(std::size_t i) {
    mark_[i] = stamp_;
  }

  // Return true if i was already visited; otherwise mark and return false.
  bool test_and_set(std::size_t i) {
    if (mark_[i] == stamp_) return true;
    mark_[i] = stamp_;
    return false;
  }

 private:
  void ensure_size(std::size_t n) {
    if (mark_.size() < n) mark_.resize(n, 0);
  }

  void advance_stamp() {
    // stamp_ starts from 1
    ++stamp_;
    if (stamp_ == 0) {
      // overflow: clear all marks
      std::fill(mark_.begin(), mark_.end(), 0);
      stamp_ = 1;
    }
  }

  std::vector<uint32_t> mark_;
  uint32_t stamp_ = 1;
};

}  // namespace vecdb
