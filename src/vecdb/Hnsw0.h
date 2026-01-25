#pragma once

#include <cstddef>
#include <vector>

#include "Distance.h"
#include "VectorStore.h"
#include "SearchResult.h"
#include "Visited.h"

namespace vecdb {

class Hnsw0 {
 public:
  struct Params {
    std::size_t M = 16;
    std::size_t ef_construction = 100;
    bool use_diversity = true;
  };

  Hnsw0(const VectorStore& store, Metric metric)
      : store_(store), metric_(metric), params_() {}

  Hnsw0(const VectorStore& store, Metric metric, Params params)
      : store_(store), metric_(metric), params_(params) {}

  void insert(std::size_t index);

  std::vector<SearchResult> search(const std::vector<float>& query,
                                   std::size_t k,
                                   std::size_t ef_search) const;

  bool empty() const { return !has_entry_; }
  std::size_t size() const { return neighbors_.size(); }

 private:
  std::vector<SearchResult> search_layer0(const float* query_ptr,
                                         std::size_t entry,
                                         std::size_t ef_search) const;

  std::vector<std::size_t> select_neighbors_simple(const std::vector<SearchResult>& candidates,
                                                   std::size_t M) const;

  std::vector<std::size_t> select_neighbors_diverse(std::size_t base,
                                                    const std::vector<SearchResult>& candidates,
                                                    std::size_t M) const;

  void prune_neighbors(std::size_t node);
  void connect_bidirectional(std::size_t u, std::size_t v);

  const VectorStore& store_;
  Metric metric_;
  Params params_;

  std::vector<std::vector<std::size_t>> neighbors_;

  std::size_t entry_point_ = 0;
  bool has_entry_ = false;

  // Reusable visited buffer for searches (stamp-array).
  mutable Visited visited_;
};

}  // namespace vecdb
