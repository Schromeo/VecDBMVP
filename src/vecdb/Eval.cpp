#include "Eval.h"

#include <algorithm>
#include <chrono>
#include <unordered_set>

namespace vecdb {

double Evaluator::recall_at_k(const std::vector<SearchResult>& truth,
                              const std::vector<SearchResult>& approx,
                              std::size_t k) {
  if (k == 0) return 0.0;

  std::size_t kt = std::min(k, truth.size());
  std::size_t ka = std::min(k, approx.size());

  // Put truth topK indices into a set
  std::unordered_set<std::size_t> truth_set;
  truth_set.reserve(kt * 2);

  for (std::size_t i = 0; i < kt; ++i) {
    truth_set.insert(truth[i].index);
  }

  // Count how many approx topK are in truth topK
  std::size_t hit = 0;
  for (std::size_t i = 0; i < ka; ++i) {
    if (truth_set.find(approx[i].index) != truth_set.end()) {
      ++hit;
    }
  }

  // recall@k = hits / k (common definition), but if truth has fewer than k,
  // we normalize by kt to avoid penalizing small datasets.
  return kt == 0 ? 0.0 : static_cast<double>(hit) / static_cast<double>(kt);
}

EvalReport Evaluator::evaluate(const std::vector<std::vector<float>>& queries,
                               std::size_t k,
                               const SearchFn& truth,
                               const SearchFn& approx) const {
  using clock = std::chrono::steady_clock;

  double total_recall = 0.0;
  double total_ms = 0.0;

  for (const auto& q : queries) {
    // ground truth
    auto gt = truth(q, k);

    // approx timing
    auto t0 = clock::now();
    auto ap = approx(q, k);
    auto t1 = clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms += ms;

    total_recall += recall_at_k(gt, ap, k);
  }

  EvalReport r;
  if (!queries.empty()) {
    r.recall_at_k = total_recall / static_cast<double>(queries.size());
    r.avg_latency_ms = total_ms / static_cast<double>(queries.size());
  }
  return r;
}

}  // namespace vecdb
