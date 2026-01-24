#include "Hnsw0.h"

#include <algorithm>
#include <queue>
#include <unordered_set>
#include <stdexcept>
#include <limits>

namespace vecdb {

namespace {

// For best-first expansion: smaller distance first
struct Cand {
  std::size_t index;
  float dist;
};

struct CandMinHeap {
  bool operator()(const Cand& a, const Cand& b) const {
    return a.dist > b.dist;  // makes priority_queue a min-heap
  }
};

// For maintaining best ef results: keep worst on top (max-heap)
struct ResMaxHeap {
  bool operator()(const Cand& a, const Cand& b) const {
    return a.dist < b.dist;  // larger dist has higher priority
  }
};

}  // namespace

std::vector<SearchResult> Hnsw0::search_layer0(const float* query_ptr,
                                              std::size_t entry,
                                              std::size_t ef_search) const {
  if (!has_entry_ || ef_search == 0) return {};

  auto dist_to = [&](std::size_t idx) -> float {
    const float* v = store_.get_ptr(idx);
    // store_.get_ptr already checks alive; but be defensive
    if (!v) return std::numeric_limits<float>::infinity();
    return Distance::distance(metric_, query_ptr, v, store_.dim());
  };

  std::unordered_set<std::size_t> visited;
  visited.reserve(ef_search * 8);

  float entry_d = dist_to(entry);

  std::priority_queue<Cand, std::vector<Cand>, CandMinHeap> candidates;
  std::priority_queue<Cand, std::vector<Cand>, ResMaxHeap> results;

  candidates.push({entry, entry_d});
  results.push({entry, entry_d});
  visited.insert(entry);

  while (!candidates.empty()) {
    Cand c = candidates.top();
    candidates.pop();

    // Stop condition: best candidate is worse than worst in results
    Cand worst = results.top();
    if (c.dist > worst.dist) break;

    for (std::size_t nb : neighbors_[c.index]) {
      if (!store_.is_alive(nb)) continue;
      if (visited.find(nb) != visited.end()) continue;
      visited.insert(nb);

      float d = dist_to(nb);

      if (results.size() < ef_search) {
        candidates.push({nb, d});
        results.push({nb, d});
      } else if (d < results.top().dist) {
        candidates.push({nb, d});
        results.push({nb, d});
        if (results.size() > ef_search) results.pop();
      }
    }
  }

  std::vector<SearchResult> out;
  out.reserve(results.size());
  while (!results.empty()) {
    auto x = results.top();
    results.pop();
    out.push_back({x.index, x.dist});
  }

  std::sort(out.begin(), out.end(),
            [](const SearchResult& a, const SearchResult& b) {
              return a.distance < b.distance;
            });
  return out;
}

std::vector<std::size_t> Hnsw0::select_neighbors_simple(const std::vector<SearchResult>& candidates,
                                                        std::size_t M) const {
  std::vector<std::size_t> out;
  out.reserve(std::min(M, candidates.size()));
  for (std::size_t i = 0; i < candidates.size() && out.size() < M; ++i) {
    out.push_back(candidates[i].index);
  }
  return out;
}

void Hnsw0::prune_neighbors(std::size_t node) {
  auto& nbrs = neighbors_[node];
  if (nbrs.size() <= params_.M) return;

  const float* base = store_.get_ptr(node);
  if (!base) return;

  std::vector<std::pair<float, std::size_t>> scored;
  scored.reserve(nbrs.size());

  for (auto nb : nbrs) {
    const float* v = store_.get_ptr(nb);
    if (!v) continue;
    float d = Distance::distance(metric_, base, v, store_.dim());
    scored.push_back({d, nb});
  }

  std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  nbrs.clear();
  nbrs.reserve(std::min(params_.M, scored.size()));
  for (std::size_t i = 0; i < scored.size() && nbrs.size() < params_.M; ++i) {
    nbrs.push_back(scored[i].second);
  }
}

void Hnsw0::connect_bidirectional(std::size_t u, std::size_t v) {
  neighbors_[u].push_back(v);
  neighbors_[v].push_back(u);
  prune_neighbors(u);
  prune_neighbors(v);
}

void Hnsw0::insert(std::size_t index) {
  if (!store_.is_alive(index)) return;

  if (index >= neighbors_.size()) {
    neighbors_.resize(index + 1);
  }

  // First node becomes entry point
  if (!has_entry_) {
    entry_point_ = index;
    has_entry_ = true;
    return;
  }

  // Search candidates using the new node as "query" without copying:
  const float* q = store_.get_ptr(index);
  if (!q) return;

  auto candidates = search_layer0(q, entry_point_, params_.ef_construction);

  // Remove itself if present (possible if graph contains it already; defensive)
  candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                  [&](const SearchResult& r) { return r.index == index; }),
                   candidates.end());

  // Choose up to M closest neighbors
  auto chosen = select_neighbors_simple(candidates, params_.M);

  // Connect bidirectionally
  for (auto nb : chosen) {
    if (nb >= neighbors_.size()) neighbors_.resize(nb + 1);
    connect_bidirectional(index, nb);
  }

  // Keep entry point fixed for MVP.
  // (Future: choose highest-level node, or update by heuristic.)
}

std::vector<SearchResult> Hnsw0::search(const std::vector<float>& query,
                                       std::size_t k,
                                       std::size_t ef_search) const {
  if (!has_entry_ || k == 0) return {};
  if (query.size() != store_.dim()) {
    throw std::invalid_argument("Hnsw0::search: query dim mismatch");
  }

  std::size_t ef = std::max(ef_search, k);
  auto candidates = search_layer0(query.data(), entry_point_, ef);

  if (candidates.size() > k) candidates.resize(k);
  return candidates;
}

}  // namespace vecdb
