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
    if (i + 1 != v.size()) std::cout << ", ";
  }
  std::cout << "]";
}

int main() {
  std::cout << "VecDB MVP starting..." << std::endl;

#ifdef _WIN32
  std::cout << "Platform: Windows" << std::endl;
#elif __APPLE__
  std::cout << "Platform: macOS" << std::endl;
#elif __linux__
  std::cout << "Platform: Linux" << std::endl;
#else
  std::cout << "Platform: Unknown" << std::endl;
#endif

  std::cout << std::fixed << std::setprecision(6);

  // ---------------- Distance sanity checks ----------------
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

    float l2_ab = Distance::distance(Metric::L2, a.data(), b.data(), a.size());
    float l2_ac = Distance::distance(Metric::L2, a.data(), c.data(), a.size());
    std::cout << "L2^2(a,b) = " << l2_ab << "  (expected 1)\n";
    std::cout << "L2^2(a,c) = " << l2_ac << "  (expected 2)\n";

    float cos_ab = Distance::distance(Metric::COSINE, a.data(), b.data(), a.size());
    float cos_ac = Distance::distance(Metric::COSINE, a.data(), c.data(), a.size());
    std::cout << "cosDist(a,b) = " << cos_ab << "  (expected 0, same direction)\n";
    std::cout << "cosDist(a,c) = " << cos_ac << "  (expected 1, orthogonal)\n";

    std::vector<float> d{3.0f, 4.0f};  // norm 5
    Distance::normalize_inplace(d.data(), d.size());
    std::cout << "normalize([3,4]) = "; print_vec(d);
    std::cout << "  (expected [0.6,0.8])\n";
  }

  // ---------------- VectorStore sanity checks ----------------
  {
    std::cout << "\nVectorStore sanity checks:\n";
    vecdb::VectorStore store(/*dim=*/3);

    std::size_t i0 = store.insert("u1", std::vector<float>{1, 2, 3});
    std::size_t i1 = store.insert("u2", std::vector<float>{2, 3, 4});

    std::cout << "insert u1 -> index " << i0 << "\n";
    std::cout << "insert u2 -> index " << i1 << "\n";
    std::cout << "store.size = " << store.size() << " (expected 2)\n";

    const float* p = store.get_ptr(i0);
    std::cout << "get_ptr(u1) = " << (p ? "OK" : "nullptr")
              << "  first=" << (p ? p[0] : -1) << "\n";

    bool removed = store.remove("u1");
    std::cout << "remove(u1) = " << (removed ? "true" : "false") << " (expected true)\n";
    std::cout << "contains(u1) = " << (store.contains("u1") ? "true" : "false") << " (expected false)\n";
    std::cout << "get_ptr(u1_index) = " << (store.get_ptr(i0) ? "OK" : "nullptr")
              << " (expected nullptr)\n";
  }

  // ---------------- Bruteforce demo ----------------
  {
    std::cout << "\nBruteforce demo:\n";
    vecdb::VectorStore s2(/*dim=*/4);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    // Insert 100 random vectors
    for (int i = 0; i < 100; ++i) {
      std::vector<float> v(4);
      for (int j = 0; j < 4; ++j) v[j] = uni(rng);
      s2.insert("id_" + std::to_string(i), v);
    }

    // Random query
    std::vector<float> q(4);
    for (int j = 0; j < 4; ++j) q[j] = uni(rng);

    vecdb::Bruteforce bf(s2, vecdb::Metric::L2);
    auto top = bf.search(q, /*k=*/5);

    std::cout << "Query q="; print_vec(q); std::cout << "\n";
    std::cout << "Top5 (L2^2):\n";
    for (const auto& r : top) {
      std::cout << "  index=" << r.index
                << " id=" << s2.id_at(r.index)
                << " dist=" << r.distance << "\n";
    }
  }

  // ---------------- Eval + HNSW0 demo ----------------
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
      for (std::size_t j = 0; j < dim; ++j) v[j] = uni(rng);
      store.insert("pt_" + std::to_string(i), v);
    }

    std::vector<std::vector<float>> queries;
    queries.reserve(num_queries);
    for (std::size_t qi = 0; qi < num_queries; ++qi) {
      std::vector<float> q(dim);
      for (std::size_t j = 0; j < dim; ++j) q[j] = uni(rng);
      queries.push_back(std::move(q));
    }

    vecdb::Bruteforce truth_bf(store, vecdb::Metric::L2);

    vecdb::Hnsw0::Params hp;
    hp.M = 16;
    hp.ef_construction = 100;
    vecdb::Hnsw0 hnsw(store, vecdb::Metric::L2, hp);

    // Build index
    for (std::size_t i = 0; i < store.size(); ++i) {
      if (store.is_alive(i)) hnsw.insert(i);
    }

    vecdb::Evaluator eval(store);

    const std::size_t ef_search = 50;  // try 20 / 50 / 100
    auto report = eval.evaluate(
        queries, k,
        [&](const std::vector<float>& q, std::size_t kk) { return truth_bf.search(q, kk); },
        [&](const std::vector<float>& q, std::size_t kk) { return hnsw.search(q, kk, ef_search); }
    );

    std::cout << "N=" << N << " dim=" << dim << " queries=" << num_queries
              << " k=" << k << " ef_search=" << ef_search
              << " M=" << hp.M << " efC=" << hp.ef_construction << "\n";
    std::cout << "recall@k = " << report.recall_at_k << "\n";
    std::cout << "avg_latency_ms (approx path) = " << report.avg_latency_ms << "\n";
  }

  std::cout << "\nNext: tune ef_search/M, then add hierarchical HNSW levels and diversity heuristic.\n";
  return 0;
}
