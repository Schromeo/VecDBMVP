#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "vecdb/Collection.h"
#include "vecdb/Distance.h"
#include "vecdb/Hnsw.h"
#include "vecdb/VectorStore.h"

using Clock = std::chrono::high_resolution_clock;

static void print_vec(const std::vector<float>& v, std::size_t max_elems = 8) {
  std::cout << "[";
  for (std::size_t i = 0; i < v.size() && i < max_elems; ++i) {
    if (i) std::cout << ", ";
    std::cout << std::fixed << std::setprecision(6) << v[i];
  }
  if (v.size() > max_elems) std::cout << ", ...";
  std::cout << "]";
}

static std::string platform_name() {
#if defined(_WIN32)
  return "Windows";
#elif defined(__APPLE__)
  return "macOS";
#elif defined(__linux__)
  return "Linux";
#else
  return "Unknown";
#endif
}

static std::vector<float> rand_vec(std::mt19937& rng, std::size_t dim) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (std::size_t i = 0; i < dim; ++i) v[i] = dist(rng);
  return v;
}

static std::vector<std::pair<std::size_t, float>> bruteforce_topk(
    const vecdb::VectorStore& store,
    vecdb::Metric metric,
    const std::vector<float>& query,
    std::size_t k) {

  std::vector<std::pair<std::size_t, float>> all;
  all.reserve(store.size());

  for (std::size_t i = 0; i < store.size(); ++i) {
    if (!store.is_alive(i)) continue;
    const float* p = store.get_ptr(i);
    if (!p) continue;
    float d = vecdb::Distance::distance(metric, query.data(), p, store.dim());
    all.push_back({i, d});
  }

  if (k > all.size()) k = all.size();
  std::nth_element(all.begin(), all.begin() + k, all.end(),
                   [](auto& a, auto& b) { return a.second < b.second; });
  all.resize(k);
  std::sort(all.begin(), all.end(),
            [](auto& a, auto& b) { return a.second < b.second; });
  return all;
}

static double recall_at_k(const std::vector<std::vector<std::size_t>>& truth,
                          const std::vector<std::vector<std::size_t>>& approx) {
  std::size_t hit = 0;
  std::size_t total = 0;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    std::unordered_set<std::size_t> S(truth[i].begin(), truth[i].end());
    for (auto x : approx[i]) if (S.count(x)) ++hit;
    total += truth[i].size();
  }
  return total ? static_cast<double>(hit) / static_cast<double>(total) : 0.0;
}

static void run_hnsw_benchmark() {
  std::mt19937 rng(123);

  const std::size_t N = 200000;     // 先别用 500k，避免你本机卡到飞起；你想冲再改大
  const std::size_t dim = 32;
  const std::size_t queries = 200;
  const std::size_t k = 10;
  const std::vector<std::size_t> ef_list{10, 20, 50, 100, 200};

  vecdb::VectorStore store(dim);
  for (std::size_t i = 0; i < N; ++i) store.upsert("id_" + std::to_string(i), rand_vec(rng, dim));

  std::vector<std::vector<float>> Q;
  Q.reserve(queries);
  for (std::size_t i = 0; i < queries; ++i) Q.push_back(rand_vec(rng, dim));

  std::vector<std::vector<std::size_t>> truth;
  truth.reserve(queries);
  for (std::size_t qi = 0; qi < queries; ++qi) {
    auto top = bruteforce_topk(store, vecdb::Metric::L2, Q[qi], k);
    std::vector<std::size_t> ids;
    ids.reserve(top.size());
    for (auto& p : top) ids.push_back(p.first);
    truth.push_back(std::move(ids));
  }

  auto eval = [&](const char* label, vecdb::Hnsw::Params p) {
    vecdb::Hnsw hnsw(store, vecdb::Metric::L2, p);
    for (std::size_t i = 0; i < store.size(); ++i) if (store.is_alive(i)) hnsw.insert(i);

    std::cout << "\n[" << label << "] " << (p.use_diversity ? "Diversity ON" : "Diversity OFF")
              << " (Hierarchical HNSW)\n";

    std::cout << std::left
              << std::setw(15) << "ef_search"
              << std::setw(15) << "recall@k"
              << std::setw(18) << "avg_latency_ms"
              << "\n";

    for (auto ef : ef_list) {
      std::vector<std::vector<std::size_t>> approx;
      approx.reserve(queries);

      auto t0 = Clock::now();
      for (std::size_t qi = 0; qi < queries; ++qi) {
        auto res = hnsw.search(Q[qi], k, ef);
        std::vector<std::size_t> ids;
        ids.reserve(res.size());
        for (auto& r : res) ids.push_back(r.index);
        approx.push_back(std::move(ids));
      }
      auto t1 = Clock::now();

      double r = recall_at_k(truth, approx);
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(queries);

      std::cout << std::left
                << std::setw(15) << ef
                << std::setw(15) << std::fixed << std::setprecision(6) << r
                << std::setw(18) << std::fixed << std::setprecision(6) << ms
                << "\n";
    }
  };

  std::cout << "\nEval harness demo (truth=bruteforce, approx=HNSW):\n";
  std::cout << "N=" << N << " dim=" << dim << " queries=" << queries << " k=" << k << "\n";

  vecdb::Hnsw::Params A;
  A.M = 16; A.M0 = 32; A.ef_construction = 100; A.use_diversity = false;
  A.seed = 123; A.level_mult = 1.0f;

  vecdb::Hnsw::Params B = A;
  B.use_diversity = true;

  eval("A", A);
  eval("B", B);
}

static void persistence_demo() {
  namespace fs = std::filesystem;

  std::cout << "\nPersistence demo:\n";

  fs::path dir = fs::path("data") / "demo_collection";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);

  vecdb::Collection::Options opt;
  opt.dim = 4;
  opt.metric = vecdb::Metric::L2;
  opt.hnsw_params.M = 16;
  opt.hnsw_params.M0 = 32;
  opt.hnsw_params.ef_construction = 100;
  opt.hnsw_params.use_diversity = true;
  opt.hnsw_params.seed = 123;
  opt.hnsw_params.level_mult = 1.0f;

  auto col = vecdb::Collection::create(dir.string(), opt);

  col.upsert("u1", std::vector<float>{1, 0, 0, 0});
  col.upsert("u2", std::vector<float>{0, 1, 0, 0});
  col.upsert("u3", std::vector<float>{0, 0, 1, 0});
  col.upsert("u4", std::vector<float>{0, 0, 0, 1});

  col.build_index();
  col.save();

  auto col2 = vecdb::Collection::open(dir.string());

  std::vector<float> q{0.9f, 0.1f, 0.f, 0.f};
  auto res = col2.search(q, /*k=*/3, /*ef_search=*/50);

  std::cout << "Reloaded collection search q=";
  print_vec(q);
  std::cout << "\nTop3:\n";
  for (auto& r : res) {
    std::cout << "  index=" << r.index
              << " id=" << col2.id_at(r.index)
              << " dist=" << std::fixed << std::setprecision(6) << r.distance
              << "\n";
  }
}

int main() {
  std::cout << std::fixed << std::setprecision(6);

  std::cout << "VecDB MVP starting...\n";
  std::cout << "Platform: " << platform_name() << "\n";

  // Distance sanity checks
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

    float cd_ab = Distance::distance(Metric::COSINE, a.data(), b.data(), a.size());
    float cd_ac = Distance::distance(Metric::COSINE, a.data(), c.data(), a.size());
    std::cout << "cosDist(a,b) = " << cd_ab << "  (expected 0, same direction)\n";
    std::cout << "cosDist(a,c) = " << cd_ac << "  (expected 1, orthogonal)\n";

    std::vector<float> x{3.0f, 4.0f};
    vecdb::Distance::normalize_inplace(x.data(), x.size());
    std::cout << "normalize([3,4]) = ";
    print_vec(x);
    std::cout << "  (expected [0.6,0.8])\n";
  }

  // VectorStore sanity checks
  {
    vecdb::VectorStore store(2);
    std::cout << "\nVectorStore sanity checks:\n";

    std::size_t i1 = store.upsert("u1", std::vector<float>{1.0f, 2.0f});
    std::cout << "insert u1 -> index " << i1 << "\n";
    std::size_t i2 = store.upsert("u2", std::vector<float>{3.0f, 4.0f});
    std::cout << "insert u2 -> index " << i2 << "\n";

    std::cout << "store.size = " << store.size() << " (expected 2)\n";
    const float* p = store.get_ptr("u1");
    std::cout << "get_ptr(u1) = " << (p ? "OK" : "nullptr");
    if (p) std::cout << "  first=" << p[0];
    std::cout << "\n";

    bool rm = store.remove("u1");
    std::cout << "remove(u1) = " << (rm ? "true" : "false") << " (expected true)\n";
    std::cout << "contains(u1) = " << (store.contains("u1") ? "true" : "false") << " (expected false)\n";

    const float* p2 = store.get_ptr(i1);
    std::cout << "get_ptr(u1_index) = " << (p2 ? "not-null" : "nullptr") << " (expected nullptr)\n";
  }

  // Bruteforce demo
  {
    std::cout << "\nBruteforce demo:\n";
    std::mt19937 rng(123);
    const std::size_t N = 100;
    const std::size_t dim = 4;

    vecdb::VectorStore store(dim);
    for (std::size_t i = 0; i < N; ++i) store.upsert("id_" + std::to_string(i), rand_vec(rng, dim));

    std::vector<float> q = rand_vec(rng, dim);
    std::cout << "Query q=";
    print_vec(q);
    std::cout << "\nTop5 (L2^2):\n";

    auto top5 = bruteforce_topk(store, vecdb::Metric::L2, q, 5);
    for (auto& it : top5) {
      std::cout << "  index=" << it.first
                << " id=id_" << it.first
                << " dist=" << it.second << "\n";
    }
  }

  // Benchmark (optional)
  run_hnsw_benchmark();

  // Persistence demo (must-have)
  persistence_demo();

  std::cout << "\nNext: add unit/integration tests + README design doc updates.\n";
  return 0;
}
