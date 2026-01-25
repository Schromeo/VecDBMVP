#include "Hnsw0.h"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <limits>

namespace vecdb {

namespace {

struct Cand {
  std::size_t index;
  float dist;
};

struct CandMinHeap {
  bool operator()(const Cand& a, const Cand& b) const { return a.dist > b.dist; }
};

struct ResMaxHeap {
  bool operator()(const Cand& a, const Cand& b) const { return a.dist < b.dist; }
};

}  // namespace

std::vector<SearchResult> Hnsw0::search_layer0(const float* query_ptr,
                                              std::size_t entry,
                                              std::size_t ef_search) const {
  if (!has_entry_ || ef_search == 0) return {};
  if (!store_.is_alive(entry)) return {};

  auto dist_to = [&](std::size_t idx) -> float {
    const float* v = store_.get_ptr(idx);
    if (!v) return std::numeric_limits<float>::infinity();
    return Distance::distance(metric_, query_ptr, v, store_.dim());
  };

  // --- visited: stamp-array ---
  visited_.start(store_.size());

  float entry_d = dist_to(entry);

  std::priority_queue<Cand, std::vector<Cand>, CandMinHeap> candidates;
  std::priority_queue<Cand, std::vector<Cand>, ResMaxHeap> results;

  candidates.push({entry, entry_d});
  results.push({entry, entry_d});
  visited_.set(entry);

  while (!candidates.empty()) {
    Cand c = candidates.top();
    candidates.pop();

    Cand worst = results.top();
    if (c.dist > worst.dist) break;

    for (std::size_t nb : neighbors_[c.index]) {
      if (!store_.is_alive(nb)) continue;
      if (visited_.test_and_set(nb)) continue;

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

std::vector<std::size_t> Hnsw0::select_neighbors_diverse(std::size_t base,
                                                         const std::vector<SearchResult>& candidates,
                                                         std::size_t M) const {
  std::vector<std::size_t> selected;
  selected.reserve(std::min(M, candidates.size()));

  const float* base_ptr = store_.get_ptr(base);
  if (!base_ptr) return selected;

  for (const auto& cand : candidates) {
    if (selected.size() >= M) break;

    std::size_t c = cand.index;
    if (!store_.is_alive(c) || c == base) continue;

    const float* c_ptr = store_.get_ptr(c);
    if (!c_ptr) continue;

    float dc_base = cand.distance;

    bool ok = true;
    for (std::size_t s : selected) {
      const float* s_ptr = store_.get_ptr(s);
      if (!s_ptr) continue;

      float dc_s = Distance::distance(metric_, c_ptr, s_ptr, store_.dim());
      if (dc_s < dc_base) {
        ok = false;
        break;
      }
    }

    if (ok) selected.push_back(c);
  }

  if (selected.size() < M) {
    for (const auto& cand : candidates) {
      if (selected.size() >= M) break;
      std::size_t c = cand.index;
      if (!store_.is_alive(c) || c == base) continue;
      if (std::find(selected.begin(), selected.end(), c) != selected.end()) continue;
      selected.push_back(c);
    }
  }

  return selected;
}

void Hnsw0::prune_neighbors(std::size_t node) {
  auto& nbrs = neighbors_[node];
  if (nbrs.size() <= params_.M) return;

  const float* base = store_.get_ptr(node);
  if (!base) return;

  std::vector<SearchResult> cand;
  cand.reserve(nbrs.size());
  for (auto nb : nbrs) {
    const float* v = store_.get_ptr(nb);
    if (!v) continue;
    float d = Distance::distance(metric_, base, v, store_.dim());
    cand.push_back({nb, d});
  }

  std::sort(cand.begin(), cand.end(),
            [](const SearchResult& a, const SearchResult& b) {
              return a.distance < b.distance;
            });

  std::vector<std::size_t> kept =
      params_.use_diversity ? select_neighbors_diverse(node, cand, params_.M)
                            : select_neighbors_simple(cand, params_.M);

  nbrs = std::move(kept);
}

void Hnsw0::connect_bidirectional(std::size_t u, std::size_t v) {
  neighbors_[u].push_back(v);
  neighbors_[v].push_back(u);
  prune_neighbors(u);
  prune_neighbors(v);
}

void Hnsw0::insert(std::size_t index) {
  if (!store_.is_alive(index)) return;

  if (index >= neighbors_.size()) neighbors_.resize(index + 1);

  if (!has_entry_) {
    entry_point_ = index;
    has_entry_ = true;
    return;
  }

  const float* q = store_.get_ptr(index);
  if (!q) return;

  auto candidates = search_layer0(q, entry_point_, params_.ef_construction);

  candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                  [&](const SearchResult& r) { return r.index == index; }),
                   candidates.end());

  std::vector<std::size_t> chosen =
      params_.use_diversity ? select_neighbors_diverse(index, candidates, params_.M)
                            : select_neighbors_simple(candidates, params_.M);

  for (auto nb : chosen) {
    if (nb >= neighbors_.size()) neighbors_.resize(nb + 1);
    connect_bidirectional(index, nb);
  }
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
