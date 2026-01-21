#pragma once

#include <cstddef>
#include <vector>

#include "Distance.h"
#include "VectorStore.h"

namespace vecdb {

// One search result entry
struct SearchResult {
  std::size_t index;  // internal index in VectorStore
  float distance;     // lower is closer
};

// Exact topK search baseline (O(N * D) distance eval)
class Bruteforce {
 public:
  Bruteforce(const VectorStore& store, Metric metric)
      : store_(store), metric_(metric) {}

  // Returns up to k nearest alive vectors to query.
  // If k > number of alive vectors, returns fewer.
  std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k) const;

 private:
  const VectorStore& store_;
  Metric metric_;
};

}  // namespace vecdb
