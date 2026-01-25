#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "Distance.h"
#include "SearchResult.h"
#include "VectorStore.h"
#include "Visited.h"

namespace vecdb {

class Hnsw {
 public:
  struct Params {
    std::size_t M = 16;
    std::size_t M0 = 32;
    std::size_t ef_construction = 100;
    bool use_diversity = true;
    unsigned seed = 123;
    float level_mult = 1.0f;
  };

  // -------- Persistence export/import (v1) --------
  //
  // We export only the graph structure (levels + neighbor lists) and entry info.
  // Vectors/ids/alive flags live in VectorStore and are persisted separately.
  struct ExportNode {
    int level = -1;  // -1 means "no links / absent"
    std::vector<std::vector<std::size_t>> links;  // links[l] = neighbor indices at level l
  };

  struct Export {
    std::size_t entry_point = 0;
    bool has_entry = false;
    int max_level = -1;
    std::vector<ExportNode> nodes;  // size == store.size()
  };

  Hnsw(const VectorStore& store, Metric metric)
      : store_(store), metric_(metric), params_() {}

  Hnsw(const VectorStore& store, Metric metric, Params params)
      : store_(store), metric_(metric), params_(params) {}

  // Insert a node (by store index) into the graph.
  void insert(std::size_t index);

  // Search for k nearest neighbors (approx) with ef_search.
  std::vector<SearchResult> search(const std::vector<float>& query,
                                   std::size_t k,
                                   std::size_t ef_search) const;

  bool empty() const { return !has_entry_; }
  int max_level() const { return max_level_; }

  // Export / import the internal graph structure for persistence.
  Export export_graph() const;
  void import_graph(const Export& ex);

 private:
  struct NodeLinks {
    std::vector<std::vector<std::size_t>> links;  // links[level] -> neighbor indices
  };

  int random_level();
  std::size_t max_deg(int level) const { return (level == 0) ? params_.M0 : params_.M; }

  void ensure_node(std::size_t index);
  int node_level(std::size_t index) const;

  std::vector<SearchResult> search_level(const float* query_ptr,
                                        std::size_t entry,
                                        int level,
                                        std::size_t ef) const;

  std::size_t greedy_descent(const float* query_ptr,
                             std::size_t entry,
                             int level) const;

  std::vector<std::size_t> select_neighbors_simple(const std::vector<SearchResult>& candidates,
                                                   std::size_t M) const;

  std::vector<std::size_t> select_neighbors_diverse(std::size_t base,
                                                    const std::vector<SearchResult>& candidates,
                                                    std::size_t M) const;

  void prune_neighbors(std::size_t node, int level);
  void connect_bidirectional(std::size_t a, std::size_t b, int level);

  const VectorStore& store_;
  Metric metric_;
  Params params_;

  std::vector<NodeLinks> graph_;

  std::size_t entry_point_ = 0;
  bool has_entry_ = false;
  int max_level_ = -1;

  mutable bool rng_inited_ = false;
  mutable unsigned rng_state_ = 0;

  // Reusable visited buffer (stamp-array) for searches.
  mutable Visited visited_;
};

}  // namespace vecdb
