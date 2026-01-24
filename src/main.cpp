#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <string>

#include "vecdb/Distance.h"
#include "vecdb/VectorStore.h"
#include "vecdb/Bruteforce.h"
#include "vecdb/Eval.h"
#include "vecdb/Hnsw0.h"

static void print_vec(const std::vector<float>& v) {
  std::cout << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i];
    if (i + 1 < v.size()) std::cout << ", ";
  }
  std::cout << "]";
}

int main() {
  std::cout << "VecDB MVP starting...\n";

#ifdef _WIN32
  std::cout << "Platform: Windows\n";
#elif __APPLE__
  std::cout << "Platform: macOS\n";
#else
  std::cout << "Platform: Unknown\n";
#endif

  std::cout << std::fixed << std::setprecision(6);

  // ------------------------------------------------------------
  // Distance sanity checks
  // ------------------------------------------------------------
  {
    using vecdb::Distance;
    using vecdb::Metric;

    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{2.0f, 0.0f};
    std::vector<float> c{0.0f, 1.0f};

    std::cout << "\nDistance sanity checks:\n";
    std::cout << "a="; print_vec(a);
    std::cout << "  b="; print_vec(b);
    std::cout << "  c="; print_vec(c);
    std::cout << "\n";

    std::cout << "L2^2(a,b) = "
              << Distance::distance(Metric::L2, a.data(), b.data(), a.size())
              << "  (expected 1)\n";
    std::cout << "L2^2(a,c) = "
              << Distance::distance(Metric::L2, a.data(), c.data(), a.size())
              << "  (expected 2)\n";

    std::cout << "cosDist(a,b) = "
              << Distance::distance(Metric::COSINE, a.data(), b.data(), a.size())
              << "  (expected 0, same direction)\n";
    std::cout << "cosDist(a,c) = "
              << Distance::distance(Metric::COSINE, a.data(), c.data(), a.size())
              << "  (expected 1, orthogonal)\n";

    std::vector<float> d{3.0f, 4.0f};
    Distance::normalize_inplace(d.data(), d.size());
    std::cout << "normalize([3,4]) = ";
    print_vec(d);
    std::cout << "  (expected [0.6,0.8])\n";
  }

  // ------------------------------------------------------------
  // VectorStore sanity checks
  // ------------------------------------------------------------
  {
    std::cout << "\nVectorStore sanity checks:\n";
    vecdb::VectorStore store(3);

    std::size_t i0 = store.insert("u1", {1, 2, 3});
    std::size_t i1 = store.insert("u2", {2, 3, 4});

    std::cout << "insert u1 -> index " << i0 << "\n";
    std::cout << "insert u2 -> index " << i1 << "\n";
    std::cout << "store.size = " << store.size() << " (expected 2)\n";

    const float* p = store.get_ptr(i0);
    std::cout << "get_ptr(u1) = " << (p ? "OK" : "nullptr")
              << "  first=" << (p ? p[0] : -1) << "\n";

    std::cout << "remove(u1) = "
              << (store.remove("u1") ? "true" : "false")
              << " (expected true)\n";
    std::cout << "contains(u1) = "
              << (store.contains("u1") ? "true" : "false")
              << " (expected false)\n";
    std::cout << "get_ptr(u1_index) = "
              << (store.get_ptr(i0) ? "OK" : "nullptr")
              << " (expected nullptr)\n";
  }

  // ------------------------------------------------------------
  // Bruteforce demo
  // ------------------------------------------------------------
  {
    std::cout << "\nBruteforce demo:\n";
    vecdb::VectorStore store(4);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    for (int i = 0; i < 100; ++i) {
      std::vector<float> v(4);
      for (float& x : v) x = uni(rng);
      store.insert("id_" + std::to_string(i), v);
    }

    std::vector<float> q(4);
    for (float& x : q) x = uni(rng);

    vecdb::Bruteforce bf(store, vecdb::Metric::L2);
    auto top = bf.search(q, 5);

    std::cout << "Query q=";
    print_vec(q);
    std::cout << "\nTop5 (L2^2):\n";
    for (auto& r : top) {
      std::cout << "  index=" << r.index
                << " id=" << store.id_at(r.index)
                << " dist=" << r.distance << "\n";
    }
  }

  // ------------------------------------------------------------
  // Eval + HNSW0 ef_search sweep
  // ------------------------------------------------------------
  {
    std::cout << "\nEval harness demo (truth=bruteforce, approx=HNSW0):\n";

    const std::size_t dim = 32;
    const std::size_t N = 5000;
    const std::size_t num_queries = 200;
    const std::size_t k = 10;

    vecdb::VectorStore store(dim);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    for (std::size_t i = 0; i < N; ++i) {
      std::vector<float> v(dim);
      for (float& x : v) x = uni(rng);
      store.insert("pt_" + std::to_string(i), v);
    }

    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (auto& q : queries)
      for (float& x : q) x = uni(rng);

    vecdb::Bruteforce truth(store, vecdb::Metric::L2);

    vecdb::Hnsw0::Params hp;
    hp.M = 16;
    hp.ef_construction = 100;
    vecdb::Hnsw0 hnsw(store, vecdb::Metric::L2, hp);

    for (std::size_t i = 0; i < store.size(); ++i)
      if (store.is_alive(i)) hnsw.insert(i);

    vecdb::Evaluator eval(store);

    std::vector<std::size_t> ef_values = {10, 20, 50, 100, 200};

    std::cout << "N=" << N
              << " dim=" << dim
              << " queries=" << num_queries
              << " k=" << k
              << " M=" << hp.M
              << " efC=" << hp.ef_construction << "\n\n";

    std::cout << "ef_search\trecall@k\tavg_latency_ms\n";

    for (std::size_t ef : ef_values) {
      auto report = eval.evaluate(
          queries, k,
          [&](const std::vector<float>& q, std::size_t kk) {
            return truth.search(q, kk);
          },
          [&](const std::vector<float>& q, std::size_t kk) {
            return hnsw.search(q, kk, ef);
          });

      std::cout << ef << "\t\t"
                << report.recall_at_k << "\t\t"
                << report.avg_latency_ms << "\n";
    }
  }

  std::cout << "\nNext: neighbor diversity heuristic, then hierarchical HNSW.\n";
  return 0;
}
