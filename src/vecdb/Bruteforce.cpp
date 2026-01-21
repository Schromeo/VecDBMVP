#include "Bruteforce.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace vecdb {

// We keep a max-heap of size k:
// - heap top is the WORST (largest distance) among current best k
// - when we find a better point, we replace the worst
struct HeapEntry {
  std::size_t index;
  float distance;
};

struct WorseFirst {
  bool operator()(const HeapEntry& a, const HeapEntry& b) const {
    // priority_queue puts "largest" on top by default using comparator,
    // here we want larger distance to be considered "higher priority".
    return a.distance < b.distance;
  }
};

std::vector<SearchResult> Bruteforce::search(const std::vector<float>& query, std::size_t k) const {
  if (query.size() != store_.dim()) {
    throw std::invalid_argument("Bruteforce::search: query dim mismatch");
  }
  if (k == 0) return {};

  std::priority_queue<HeapEntry, std::vector<HeapEntry>, WorseFirst> heap;

  for (std::size_t i = 0; i < store_.size(); ++i) {
    if (!store_.is_alive(i)) continue;
    const float* v = store_.get_ptr(i);
    if (!v) continue;

    float d = Distance::distance(metric_, query.data(), v, store_.dim());

    if (heap.size() < k) {
      heap.push({i, d});
    } else if (d < heap.top().distance) {
      heap.pop();
      heap.push({i, d});
    }
  }

  // Extract heap to vector (currently unordered), then sort ascending by distance
  std::vector<SearchResult> results;
  results.reserve(heap.size());
  while (!heap.empty()) {
    auto e = heap.top();
    heap.pop();
    results.push_back({e.index, e.distance});
  }

  std::sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) {
              return a.distance < b.distance;
            });

  return results;
}

}  // namespace vecdb
