#pragma once

#include <cstddef>
#include <vector>

#include "Distance.h"
#include "VectorStore.h"
#include "SearchResult.h"

namespace vecdb {

// Hierarchical HNSW (multi-level).
// Memory model: each node stores adjacency lists only up to its own level.
class Hnsw {
 public:
  struct Params {
    std::size_t M = 16;                  // max degree for upper layers (>=1)
    std::size_t M0 = 32;                 // max degree for layer 0 (often 2*M)
    std::size_t ef_construction = 100;   // candidate pool size during insertion
    bool use_diversity = true;           // neighbor diversity heuristic
    unsigned seed = 123;                 // RNG seed for random_level
    float level_mult = 1.0f;             // controls geometric distribution; 1.0 is fine for MVP
  };

  Hnsw(const VectorStore& store, Metric metric)
      : store_(store), metric_(metric), params_() {}

  Hnsw(const VectorStore& store, Metric metric, Params params)
      : store_(store), metric_(metric), params_(params) {}

  void insert(std::size_t index);

  // Query: returns top-k approximate nearest neighbors at layer 0.
  std::vector<SearchResult> search(const std::vector<float>& query,
                                   std::size_t k,
                                   std::size_t ef_search) const;

  bool empty() const { return !has_entry_; }
  int max_level() const { return max_level_; }

 private:
  struct NodeLinks {
    // links[l] is adjacency list at level l, where l in [0..node_level]
    std::vector<std::vector<std::size_t>> links;
  };

  // Random level generator (geometric-like)
  int random_level();

  std::size_t max_deg(int level) const { return (level == 0) ? params_.M0 : params_.M; }

  // Ensure internal arrays can address node index
  void ensure_node(std::size_t index);

  // Return node level, or -1 if node doesn't exist in graph
  int node_level(std::size_t index) const;

  // Core HNSW search on a specific level:
  // - starts from entry point
  // - returns up to ef results sorted ascending by distance
  std::vector<SearchResult> search_level(const float* query_ptr,
                                        std::size_t entry,
                                        int level,
                                        std::size_t ef) const;

  // Greedy descent step: run ef=1 on a level, return best node index
  std::size_t greedy_descent(const float* query_ptr,
                             std::size_t entry,
                             int level) const;

  // Neighbor selection
  std::vector<std::size_t> select_neighbors_simple(const std::vector<SearchResult>& candidates,
                                                   std::size_t M) const;

  std::vector<std::size_t> select_neighbors_diverse(std::size_t base,
                                                    const std::vector<SearchResult>& candidates,
                                                    std::size_t M) const;

  // Prune a node's neighbor list at a given level to degree <= max_deg(level)
  void prune_neighbors(std::size_t node, int level);

  // Add undirected edge on a given level and prune both sides
  void connect_bidirectional(std::size_t a, std::size_t b, int level);

  const VectorStore& store_;
  Metric metric_;
  Params params_;

  std::vector<NodeLinks> graph_;   // per-node adjacency up to node level

  std::size_t entry_point_ = 0;
  bool has_entry_ = false;
  int max_level_ = -1;

  // RNG state
  mutable bool rng_inited_ = false;
  mutable unsigned rng_state_ = 0;
};

}  // namespace vecdb
