#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>

#include "SearchResult.h"
#include "VectorStore.h"

namespace vecdb {

// Search function type: given query + k -> results (sorted ascending by distance)
using SearchFn = std::function<std::vector<SearchResult>(const std::vector<float>&, std::size_t)>;

struct EvalConfig {
  std::size_t k = 10;
  std::size_t num_queries = 100;
};

struct EvalReport {
  double recall_at_k = 0.0;     // in [0, 1]
  double avg_latency_ms = 0.0;  // average per query
};

class Evaluator {
 public:
  Evaluator(const VectorStore& store) : store_(store) {}

  // Evaluate an approximate search function against brute-force ground truth.
  // - truth: typically Bruteforce(store, metric).search
  // - approx: HNSW search (later). For now we can pass brute-force to validate the harness.
  EvalReport evaluate(const std::vector<std::vector<float>>& queries,
                      std::size_t k,
                      const SearchFn& truth,
                      const SearchFn& approx) const;

  // Utility: compute recall@k for a single query result set
  static double recall_at_k(const std::vector<SearchResult>& truth,
                            const std::vector<SearchResult>& approx,
                            std::size_t k);

 private:
  const VectorStore& store_;
};

}  // namespace vecdb
