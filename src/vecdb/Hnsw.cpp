#include "Hnsw.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <limits>

namespace vecdb {

namespace {

struct Cand {
  std::size_t index;
  float dist;
};

struct MinHeap {
  bool operator()(const Cand& a, const Cand& b) const { return a.dist > b.dist; }
};

struct MaxHeap {
  bool operator()(const Cand& a, const Cand& b) const { return a.dist < b.dist; }
};

static inline unsigned lcg_next(unsigned& state) {
  state = state * 1664525u + 1013904223u;
  return state;
}

static inline float lcg_uniform01(unsigned& state) {
  unsigned x = lcg_next(state) >> 8;
  return static_cast<float>(x) / static_cast<float>(1u << 24);
}

}  // namespace

void Hnsw::ensure_node(std::size_t index) {
  if (index >= graph_.size()) graph_.resize(index + 1);
}

int Hnsw::node_level(std::size_t index) const {
  if (index >= graph_.size()) return -1;
  if (graph_[index].links.empty()) return -1;
  return static_cast<int>(graph_[index].links.size()) - 1;
}

int Hnsw::random_level() {
  if (!rng_inited_) {
    rng_state_ = params_.seed;
    rng_inited_ = true;
  }

  const float p = std::exp(-1.0f / std::max(0.0001f, params_.level_mult));
  int lvl = 0;
  while (lcg_uniform01(rng_state_) < p) {
    ++lvl;
    if (lvl > 64) break;
  }
  return lvl;
}

std::vector<SearchResult> Hnsw::search_level(const float* query_ptr,
                                            std::size_t entry,
                                            int level,
                                            std::size_t ef) const {
  if (!has_entry_ || ef == 0) return {};
  if (!store_.is_alive(entry)) return {};

  auto dist_to = [&](std::size_t idx) -> float {
    const float* v = store_.get_ptr(idx);
    if (!v) return std::numeric_limits<float>::infinity();
    return Distance::distance(metric_, query_ptr, v, store_.dim());
  };

  // --- visited: stamp-array ---
  visited_.start(store_.size());

  float entry_d = dist_to(entry);

  std::priority_queue<Cand, std::vector<Cand>, MinHeap> candidates;
  std::priority_queue<Cand, std::vector<Cand>, MaxHeap> results;

  candidates.push({entry, entry_d});
  results.push({entry, entry_d});
  visited_.set(entry);

  while (!candidates.empty()) {
    Cand c = candidates.top();
    candidates.pop();

    Cand worst = results.top();
    if (c.dist > worst.dist) break;

    int nl = node_level(c.index);
    if (nl < level) continue;

    const auto& nbrs = graph_[c.index].links[level];
    for (std::size_t nb : nbrs) {
      if (!store_.is_alive(nb)) continue;
      if (visited_.test_and_set(nb)) continue;

      float d = dist_to(nb);

      if (results.size() < ef) {
        candidates.push({nb, d});
        results.push({nb, d});
      } else if (d < results.top().dist) {
        candidates.push({nb, d});
        results.push({nb, d});
        if (results.size() > ef) results.pop();
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

std::size_t Hnsw::greedy_descent(const float* query_ptr,
                                std::size_t entry,
                                int level) const {
  auto res = search_level(query_ptr, entry, level, /*ef=*/1);
  if (res.empty()) return entry;
  return res[0].index;
}

std::vector<std::size_t> Hnsw::select_neighbors_simple(const std::vector<SearchResult>& candidates,
                                                       std::size_t M) const {
  std::vector<std::size_t> out;
  out.reserve(std::min(M, candidates.size()));
  for (std::size_t i = 0; i < candidates.size() && out.size() < M; ++i) {
    out.push_back(candidates[i].index);
  }
  return out;
}

std::vector<std::size_t> Hnsw::select_neighbors_diverse(std::size_t base,
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

void Hnsw::prune_neighbors(std::size_t node, int level) {
  int nl = node_level(node);
  if (nl < level) return;

  auto& nbrs = graph_[node].links[level];
  std::size_t M = max_deg(level);
  if (nbrs.size() <= M) return;

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
      params_.use_diversity ? select_neighbors_diverse(node, cand, M)
                            : select_neighbors_simple(cand, M);

  nbrs = std::move(kept);
}

void Hnsw::connect_bidirectional(std::size_t a, std::size_t b, int level) {
  int la = node_level(a);
  int lb = node_level(b);
  if (la < level || lb < level) return;

  graph_[a].links[level].push_back(b);
  graph_[b].links[level].push_back(a);

  prune_neighbors(a, level);
  prune_neighbors(b, level);
}

void Hnsw::insert(std::size_t index) {
  if (!store_.is_alive(index)) return;

  ensure_node(index);

  int lvl = random_level();
  graph_[index].links.resize(static_cast<std::size_t>(lvl + 1));

  if (!has_entry_) {
    entry_point_ = index;
    has_entry_ = true;
    max_level_ = lvl;
    return;
  }

  const float* q = store_.get_ptr(index);
  if (!q) return;

  std::size_t ep = entry_point_;
  for (int l = max_level_; l > lvl; --l) {
    ep = greedy_descent(q, ep, l);
  }

  for (int l = std::min(lvl, max_level_); l >= 0; --l) {
    auto candidates = search_level(q, ep, l, params_.ef_construction);

    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [&](const SearchResult& r) { return r.index == index; }),
                     candidates.end());

    std::size_t M = max_deg(l);
    std::vector<std::size_t> chosen =
        params_.use_diversity ? select_neighbors_diverse(index, candidates, M)
                              : select_neighbors_simple(candidates, M);

    for (auto nb : chosen) {
      ensure_node(nb);
      if (node_level(nb) < l) continue;
      connect_bidirectional(index, nb, l);
    }

    if (!candidates.empty()) ep = candidates[0].index;
  }

  if (lvl > max_level_) {
    max_level_ = lvl;
    entry_point_ = index;
  }
}

std::vector<SearchResult> Hnsw::search(const std::vector<float>& query,
                                      std::size_t k,
                                      std::size_t ef_search) const {
  if (!has_entry_ || k == 0) return {};
  if (query.size() != store_.dim()) {
    throw std::invalid_argument("Hnsw::search: query dim mismatch");
  }

  const float* q = query.data();

  std::size_t ep = entry_point_;
  for (int l = max_level_; l > 0; --l) {
    ep = greedy_descent(q, ep, l);
  }

  std::size_t ef = std::max<std::size_t>(ef_search, k);
  auto res = search_level(q, ep, /*level=*/0, ef);
  if (res.size() > k) res.resize(k);
  return res;
}

// ---------------- Persistence export/import ----------------

Hnsw::Export Hnsw::export_graph() const {
  Export ex;
  ex.entry_point = entry_point_;
  ex.has_entry = has_entry_;
  ex.max_level = max_level_;

  // Ensure export size matches store size (stable index universe).
  const std::size_t N = store_.size();
  ex.nodes.resize(N);

  // graph_ may be smaller if no nodes inserted; resize logic is safe.
  for (std::size_t i = 0; i < N; ++i) {
    if (i >= graph_.size() || graph_[i].links.empty()) {
      ex.nodes[i].level = -1;
      ex.nodes[i].links.clear();
      continue;
    }
    int lvl = static_cast<int>(graph_[i].links.size()) - 1;
    ex.nodes[i].level = lvl;
    ex.nodes[i].links = graph_[i].links;  // deep copy
  }
  return ex;
}

void Hnsw::import_graph(const Export& ex) {
  // Import graph structure exactly as saved.
  entry_point_ = ex.entry_point;
  has_entry_ = ex.has_entry;
  max_level_ = ex.max_level;

  // We expect exported nodes to match store.size() universe.
  const std::size_t N = store_.size();
  if (ex.nodes.size() != N) {
    throw std::runtime_error("Hnsw::import_graph: node count mismatch vs store.size()");
  }

  graph_.clear();
  graph_.resize(N);

  for (std::size_t i = 0; i < N; ++i) {
    const auto& n = ex.nodes[i];
    if (n.level < 0) {
      graph_[i].links.clear();
      continue;
    }
    // Basic validation: links size should be level+1
    if (n.links.size() != static_cast<std::size_t>(n.level + 1)) {
      throw std::runtime_error("Hnsw::import_graph: links size mismatch at node " + std::to_string(i));
    }
    graph_[i].links = n.links;
  }

  // After import, we should consider RNG state uninitialized.
  rng_inited_ = false;
}

}  // namespace vecdb
