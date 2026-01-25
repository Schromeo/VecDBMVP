#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include "vecdb/Distance.h"
#include "vecdb/VectorStore.h"
#include "vecdb/Bruteforce.h"
#include "vecdb/Hnsw.h"
#include "vecdb/Collection.h"

// ---------------- Minimal test macros ----------------
static int g_failures = 0;

#define TEST_CASE(name) \
  static void name();   \
  int main_##name = (register_test(#name, &name), 0); \
  static void name()

using TestFn = void (*)();
struct TestEntry { std::string name; TestFn fn; };
static std::vector<TestEntry>& registry() { static std::vector<TestEntry> r; return r; }
static void register_test(const char* name, TestFn fn) { registry().push_back({name, fn}); }

#define REQUIRE_TRUE(cond) do { \
  if (!(cond)) { \
    std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " REQUIRE_TRUE(" #cond ") failed\n"; \
    ++g_failures; \
    return; \
  } \
} while(0)

#define REQUIRE_FALSE(cond) REQUIRE_TRUE(!(cond))

#define REQUIRE_EQ(a,b) do { \
  auto _a = (a); auto _b = (b); \
  if (!((_a) == (_b))) { \
    std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " REQUIRE_EQ failed: " #a " vs " #b \
              << " (" << _a << " vs " << _b << ")\n"; \
    ++g_failures; \
    return; \
  } \
} while(0)

#define REQUIRE_NE(a,b) do { \
  auto _a = (a); auto _b = (b); \
  if (((_a) == (_b))) { \
    std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " REQUIRE_NE failed: " #a " == " #b \
              << " (" << _a << ")\n"; \
    ++g_failures; \
    return; \
  } \
} while(0)

static void require_near(double a, double b, double eps, const char* exprA, const char* exprB, int line) {
  if (std::fabs(a - b) > eps) {
    std::cerr << "[FAIL] " << __FILE__ << ":" << line
              << " REQUIRE_NEAR failed: " << exprA << " vs " << exprB
              << " (" << a << " vs " << b << "), eps=" << eps << "\n";
    ++g_failures;
  }
}

#define REQUIRE_NEAR(a,b,eps) do { \
  require_near((double)(a), (double)(b), (double)(eps), #a, #b, __LINE__); \
  if (g_failures) return; \
} while(0)

// ---------------- Helpers ----------------
static std::vector<float> rand_vec(std::mt19937& rng, std::size_t dim) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (std::size_t i = 0; i < dim; ++i) v[i] = dist(rng);
  return v;
}

static std::filesystem::path make_temp_dir(const std::string& name) {
  namespace fs = std::filesystem;
  auto base = fs::temp_directory_path() / "vecdb_tests";
  std::error_code ec;
  fs::create_directories(base, ec);
  auto dir = base / name;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  return dir;
}

static std::vector<std::size_t> to_indices(const std::vector<vecdb::SearchResult>& r) {
  std::vector<std::size_t> out;
  out.reserve(r.size());
  for (auto& x : r) out.push_back(x.index);
  return out;
}

static double recall_at_k(const std::vector<std::size_t>& truth,
                          const std::vector<std::size_t>& approx) {
  std::unordered_set<std::size_t> S(truth.begin(), truth.end());
  std::size_t hit = 0;
  for (auto x : approx) if (S.count(x)) ++hit;
  return truth.empty() ? 0.0 : (double)hit / (double)truth.size();
}

// ---------------- Tests ----------------

TEST_CASE(test_distance_sanity) {
  using vecdb::Metric;
  std::vector<float> a{1.f, 0.f};
  std::vector<float> b{2.f, 0.f};
  std::vector<float> c{0.f, 1.f};

  float l2_ab = vecdb::Distance::distance(Metric::L2, a.data(), b.data(), a.size());
  float l2_ac = vecdb::Distance::distance(Metric::L2, a.data(), c.data(), a.size());
  REQUIRE_NEAR(l2_ab, 1.0, 1e-6);
  REQUIRE_NEAR(l2_ac, 2.0, 1e-6);

  float cd_ab = vecdb::Distance::distance(Metric::COSINE, a.data(), b.data(), a.size());
  float cd_ac = vecdb::Distance::distance(Metric::COSINE, a.data(), c.data(), a.size());
  REQUIRE_NEAR(cd_ab, 0.0, 1e-6);
  REQUIRE_NEAR(cd_ac, 1.0, 1e-6);

  std::vector<float> x{3.f, 4.f};
  vecdb::Distance::normalize_inplace(x.data(), x.size());
  REQUIRE_NEAR(x[0], 0.6, 1e-6);
  REQUIRE_NEAR(x[1], 0.8, 1e-6);
}

TEST_CASE(test_vectorstore_basic) {
  vecdb::VectorStore store(2);

  auto i1 = store.upsert("u1", std::vector<float>{1.f, 2.f});
  auto i2 = store.upsert("u2", std::vector<float>{3.f, 4.f});
  REQUIRE_EQ(i1, (std::size_t)0);
  REQUIRE_EQ(i2, (std::size_t)1);
  REQUIRE_EQ(store.size(), (std::size_t)2);

  REQUIRE_TRUE(store.contains("u1"));
  REQUIRE_TRUE(store.contains("u2"));

  const float* p = store.get_ptr("u1");
  REQUIRE_TRUE(p != nullptr);
  REQUIRE_NEAR(p[0], 1.0, 1e-6);

  // update keeps index stable
  auto i1b = store.upsert("u1", std::vector<float>{9.f, 9.f});
  REQUIRE_EQ(i1b, i1);
  const float* p2 = store.get_ptr(i1);
  REQUIRE_TRUE(p2 != nullptr);
  REQUIRE_NEAR(p2[0], 9.0, 1e-6);

  // tombstone delete
  REQUIRE_TRUE(store.remove("u1"));
  REQUIRE_FALSE(store.contains("u1"));
  REQUIRE_FALSE(store.is_alive(i1));
  REQUIRE_TRUE(store.get_ptr(i1) == nullptr);
}

TEST_CASE(test_bruteforce_topk_matches_manual) {
  vecdb::VectorStore store(2);
  // points: (0,0), (1,0), (0,1)
  store.upsert("p0", {0.f, 0.f}); // idx 0
  store.upsert("p1", {1.f, 0.f}); // idx 1
  store.upsert("p2", {0.f, 1.f}); // idx 2

  std::vector<float> q{0.9f, 0.1f};

  vecdb::Bruteforce bf(store, vecdb::Metric::L2);
  auto top2 = bf.search(q, 2);

  REQUIRE_EQ(top2.size(), (std::size_t)2);

  // nearest should be p1 (idx 1) with dist (0.1^2+0.1^2)=0.02
  REQUIRE_EQ(top2[0].index, (std::size_t)1);
  REQUIRE_NEAR(top2[0].distance, 0.02, 1e-6);
}


TEST_CASE(test_hnsw_search_recall_small_dataset) {
  std::mt19937 rng(123);
  const std::size_t N = 2000;
  const std::size_t dim = 16;
  const std::size_t k = 10;
  const std::size_t queries = 30;

  vecdb::VectorStore store(dim);
  for (std::size_t i = 0; i < N; ++i) {
    store.upsert("id_" + std::to_string(i), rand_vec(rng, dim));
  }

  vecdb::Hnsw::Params p;
  p.M = 16;
  p.M0 = 32;
  p.ef_construction = 100;
  p.use_diversity = true;
  p.seed = 123;
  p.level_mult = 1.0f;

  vecdb::Hnsw hnsw(store, vecdb::Metric::L2, p);
  for (std::size_t i = 0; i < store.size(); ++i) {
    if (store.is_alive(i)) hnsw.insert(i);
  }

  vecdb::Bruteforce bf(store, vecdb::Metric::L2);

  double avg_recall = 0.0;
  for (std::size_t qi = 0; qi < queries; ++qi) {
    auto q = rand_vec(rng, dim);

    auto truth = bf.search(q, k);
    auto approx = hnsw.search(q, k, /*ef_search=*/200);

    auto truth_idx = to_indices(truth);
    auto approx_idx = to_indices(approx);

    avg_recall += recall_at_k(truth_idx, approx_idx);
  }
  avg_recall /= (double)queries;

  REQUIRE_TRUE(avg_recall > 0.90);
}


TEST_CASE(test_collection_persistence_roundtrip) {
  namespace fs = std::filesystem;
  auto dir = make_temp_dir("persistence_roundtrip");

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

  col.upsert("u1", {1,0,0,0});
  col.upsert("u2", {0,1,0,0});
  col.upsert("u3", {0,0,1,0});
  col.upsert("u4", {0,0,0,1});

  col.build_index();
  col.save();

  auto col2 = vecdb::Collection::open(dir.string());
  REQUIRE_TRUE(col2.has_index());

  std::vector<float> q{0.9f, 0.1f, 0.f, 0.f};
  auto res = col2.search(q, /*k=*/3, /*ef_search=*/50);
  REQUIRE_TRUE(res.size() >= 1);
  REQUIRE_EQ(col2.id_at(res[0].index), std::string("u1"));
  REQUIRE_NEAR(res[0].distance, 0.02, 1e-6);
}

// ---------------- Runner ----------------
int main() {
  std::cout << "VecDB tests starting...\n";
  for (auto& t : registry()) {
    int before = g_failures;
    t.fn();
    if (g_failures == before) {
      std::cout << "[PASS] " << t.name << "\n";
    }
  }

  if (g_failures) {
    std::cerr << "Tests finished with failures: " << g_failures << "\n";
    return 1;
  }
  std::cout << "All tests passed.\n";
  return 0;
}
