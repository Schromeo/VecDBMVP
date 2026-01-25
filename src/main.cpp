#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "vecdb/Collection.h"
#include "vecdb/Csv.h"
#include "vecdb/Distance.h"
#include "vecdb/Hnsw.h"
#include "vecdb/Metadata.h"
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

// ---------------- Simple arg parsing ----------------
struct Args {
  std::vector<std::string> pos;
  std::vector<std::pair<std::string, std::string>> kv; // --key value
  std::unordered_set<std::string> flags;               // --flag
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    if (s.rfind("--", 0) == 0) {
      // flag or key-value
      if (i + 1 < argc) {
        std::string next(argv[i + 1]);
        if (next.rfind("--", 0) != 0) {
          a.kv.push_back({s, next});
          ++i;
          continue;
        }
      }
      a.flags.insert(s);
    } else {
      a.pos.push_back(s);
    }
  }
  return a;
}

static bool has_flag(const Args& a, const std::string& k) {
  return a.flags.count(k) > 0;
}

static bool get_kv(const Args& a, const std::string& k, std::string& out) {
  for (auto& p : a.kv) {
    if (p.first == k) { out = p.second; return true; }
  }
  return false;
}

static std::size_t get_size_or(const Args& a, const std::string& k, std::size_t def) {
  std::string v;
  if (!get_kv(a, k, v)) return def;
  return static_cast<std::size_t>(std::stoull(v));
}

static int get_int_or(const Args& a, const std::string& k, int def) {
  std::string v;
  if (!get_kv(a, k, v)) return def;
  return std::stoi(v);
}

static float get_float_or(const Args& a, const std::string& k, float def) {
  std::string v;
  if (!get_kv(a, k, v)) return def;
  return std::stof(v);
}

static vecdb::Metric parse_metric(const std::string& s) {
  if (s == "l2" || s == "L2") return vecdb::Metric::L2;
  if (s == "cosine" || s == "COSINE") return vecdb::Metric::COSINE;
  throw std::invalid_argument("unknown metric: " + s + " (use l2|cosine)");
}

static bool parse_metadata_kv(const std::string& s, vecdb::Metadata& out, std::string& err) {
  return vecdb::metadata::decode(s, out, err);
}

static bool parse_filter(const Args& a, vecdb::Collection::MetadataFilter& out, std::string& err) {
  std::string s;
  if (!get_kv(a, "--filter", s)) return true;

  auto pos = s.find('=');
  if (pos == std::string::npos || pos == 0 || pos + 1 >= s.size()) {
    err = "filter must be in form key=value";
    return false;
  }
  out.key = s.substr(0, pos);
  out.value = s.substr(pos + 1);
  return true;
}

static void print_help() {
  std::cout <<
R"(VecDB MVP CLI

USAGE:
  vecdb <command> [options]

COMMANDS:
  create   Create a new collection (writes manifest/store)
  load     Load vectors from CSV into an existing collection
  build    Build HNSW index and persist it
  search   Search topK for a query (or query CSV)
  stats    Print collection info
  demo     Run built-in demo/benchmark/persistence

CSV FORMATS:
  vectors.csv: id,f1,f2,...,f_dim
  queries.csv: f1,f2,...,f_dim   OR   id,f1,...,f_dim

COMMON OPTIONS:
  --dir <path>          Collection directory (e.g., data/mycol)
  --metric l2|cosine    Metric (default l2)
  --header              CSV has a header row (skip first row)
  --has-id              CSV first column is id (even if numeric)
  --meta                CSV has a trailing metadata column

create OPTIONS:
  --dim <n>             Vector dimension (required)
  --M <n>               HNSW M (default 16)
  --M0 <n>              HNSW M0 (default 32)
  --efC <n>             HNSW ef_construction (default 100)
  --diversity 0|1       Neighbor diversity heuristic (default 1)
  --seed <n>            RNG seed (default 123)
  --level_mult <f>      Level multiplier (default 1.0)

load OPTIONS:
  --csv <file>          vectors.csv path (required)
  --build 0|1           build index after load (default 0)
  --meta                vectors.csv has trailing metadata column (key=value;key2=value2)

build OPTIONS:
  (same HNSW params as create; overrides manifest params before building)

search OPTIONS:
  --query <csvline>     Single query line: f1,f2,...,f_dim  (no id)
  --query_csv <file>    Query CSV file (multiple queries)
  --k <n>               TopK (default 10)
  --ef <n>              ef_search (default 50)
  --limit <n>           For query_csv, limit number of queries (default all)
  --filter k=v          Filter by metadata key/value (exact match)

EXAMPLES:
  vecdb create --dir data/demo --dim 768 --metric l2
  vecdb load   --dir data/demo --csv data/vectors.csv
  vecdb build  --dir data/demo --M 16 --M0 32 --efC 100 --diversity 1
  vecdb search --dir data/demo --query "0.1,0.2,0.3,..." --k 10 --ef 100
  vecdb search --dir data/demo --query_csv data/queries.csv --k 10 --ef 100

)";
}

// ---------------- Demo / benchmark (kept) ----------------

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

  const std::size_t N = 200000;
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

static int run_demo() {
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

    bool ok = store.remove("u1");
    std::cout << "remove(u1) = " << (ok ? "true" : "false") << " (expected true)\n";
    std::cout << "contains(u1) = " << (store.contains("u1") ? "true" : "false") << " (expected false)\n";
    std::cout << "get_ptr(u1_index) = " << (store.get_ptr(i1) ? "non-null" : "nullptr") << " (expected nullptr)\n";
  }

  run_hnsw_benchmark();
  persistence_demo();
  return 0;
}

// ---------------- CLI commands ----------------

static bool manifest_exists(const std::string& dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path p = fs::path(dir) / "manifest.json";
  return fs::exists(p, ec) && fs::is_regular_file(p, ec);
}

static vecdb::Hnsw::Params read_hnsw_params_from_args(const Args& a) {
  vecdb::Hnsw::Params p;
  p.M = static_cast<std::size_t>(get_size_or(a, "--M", 16));
  p.M0 = static_cast<std::size_t>(get_size_or(a, "--M0", 32));
  p.ef_construction = static_cast<std::size_t>(get_size_or(a, "--efC", 100));
  p.use_diversity = (get_int_or(a, "--diversity", 1) != 0);
  p.seed = static_cast<std::uint32_t>(get_size_or(a, "--seed", 123));
  p.level_mult = get_float_or(a, "--level_mult", 1.0f);
  return p;
}

static int cmd_create(const Args& a) {
  std::string dir;
  if (!get_kv(a, "--dir", dir)) {
    std::cerr << "create: missing --dir\n";
    return 2;
  }
  if (manifest_exists(dir)) {
    std::cerr << "create: manifest already exists in dir: " << dir << "\n";
    return 2;
  }

  std::size_t dim = get_size_or(a, "--dim", 0);
  if (dim == 0) {
    std::cerr << "create: missing --dim\n";
    return 2;
  }

  std::string metric_s = "l2";
  get_kv(a, "--metric", metric_s);

  vecdb::Collection::Options opt;
  opt.dim = dim;
  opt.metric = parse_metric(metric_s);
  opt.hnsw_params = read_hnsw_params_from_args(a);

  auto col = vecdb::Collection::create(dir, opt);
  std::cout << "Created collection at: " << col.dir()
            << " dim=" << col.dim()
            << " metric=" << metric_s
            << "\n";
  return 0;
}

static int cmd_load(const Args& a) {
  std::string dir;
  std::string csv_path;
  if (!get_kv(a, "--dir", dir)) { std::cerr << "load: missing --dir\n"; return 2; }
  if (!get_kv(a, "--csv", csv_path)) { std::cerr << "load: missing --csv\n"; return 2; }
  if (!manifest_exists(dir)) { std::cerr << "load: collection not found (manifest.json missing): " << dir << "\n"; return 2; }

  auto col = vecdb::Collection::open(dir);

  vecdb::csv::Options opt;
  opt.has_header = has_flag(a, "--header");
  opt.has_id = true;       // load requires id as first column
  opt.infer_id = false;
  opt.allow_metadata = has_flag(a, "--meta");

  std::size_t inserted = 0;
  std::string err;
  bool ok = vecdb::csv::for_each_row(csv_path, col.dim(),
    [&](const vecdb::csv::Row& row) -> bool {
      if (!row.has_id || row.id.empty()) {
        std::cerr << "load: vectors.csv must contain id as first column: id,f1,...,f_dim\n";
        return false;
      }
      vecdb::Metadata meta;
      if (opt.allow_metadata) {
        if (!row.has_metadata) {
          std::cerr << "load: --meta enabled but row has no metadata column\n";
          return false;
        }
        std::string merr;
        if (!parse_metadata_kv(row.metadata_raw, meta, merr)) {
          std::cerr << "load: metadata parse error: " << merr << "\n";
          return false;
        }
      }
      col.upsert(row.id, row.vec, meta);
      ++inserted;
      return true;
    }, err, opt);

  if (!ok) {
    std::cerr << "load failed: " << err << "\n";
    return 2;
  }

  // After loading, index is invalidated; save store + manifest (and remove hnsw.bin if existed).
  col.save();
  std::cout << "Loaded vectors: " << inserted << " into " << dir << "\n";

  int build = get_int_or(a, "--build", 0);
  if (build != 0) {
    col.build_index();
    col.save();
    std::cout << "Index built and saved.\n";
  }
  return 0;
}

static int cmd_build(const Args& a) {
  std::string dir;
  if (!get_kv(a, "--dir", dir)) { std::cerr << "build: missing --dir\n"; return 2; }
  if (!manifest_exists(dir)) { std::cerr << "build: collection not found (manifest.json missing): " << dir << "\n"; return 2; }

  auto col = vecdb::Collection::open(dir);

  // Optional overrides:
  std::string metric_s;
  if (get_kv(a, "--metric", metric_s)) {
    col.set_metric(parse_metric(metric_s));
  }
  // if any HNSW-related option present, override params
  bool has_any_param =
      get_kv(a, "--M", metric_s) || get_kv(a, "--M0", metric_s) ||
      get_kv(a, "--efC", metric_s) || get_kv(a, "--diversity", metric_s) ||
      get_kv(a, "--seed", metric_s) || get_kv(a, "--level_mult", metric_s);
  if (has_any_param) {
    col.set_hnsw_params(read_hnsw_params_from_args(a));
  }

  std::cout << "Building index for dir=" << dir << " (alive=" << col.alive_count() << ")\n";
  col.build_index();
  col.save();
  std::cout << "Index built and saved.\n";
  return 0;
}

static bool parse_query_from_string(const std::string& s,
                                    std::size_t dim,
                                    std::vector<float>& out,
                                    bool force_id) {
  vecdb::csv::Row row;
  std::string err;
  vecdb::csv::Options opt;
  opt.has_id = force_id;
  opt.infer_id = !force_id;
  if (!vecdb::csv::parse_line(s, dim, row, err, opt)) return false;
  // If user provides "id,..." for --query, ignore id.
  out = row.vec;
  return out.size() == dim;
}

static int cmd_search(const Args& a) {
  std::string dir;
  if (!get_kv(a, "--dir", dir)) { std::cerr << "search: missing --dir\n"; return 2; }
  if (!manifest_exists(dir)) { std::cerr << "search: collection not found (manifest.json missing): " << dir << "\n"; return 2; }

  std::size_t k = get_size_or(a, "--k", 10);
  std::size_t ef = get_size_or(a, "--ef", 50);
  bool has_header = has_flag(a, "--header");
  bool force_id = has_flag(a, "--has-id");

  vecdb::Collection::MetadataFilter filter;
  std::string ferr;
  if (!parse_filter(a, filter, ferr)) {
    std::cerr << "search: " << ferr << "\n";
    return 2;
  }

  auto col = vecdb::Collection::open(dir);
  if (!col.has_index() && filter.empty()) {
    std::cerr << "search: index not found. Run: vecdb build --dir " << dir << "\n";
    return 2;
  }

  std::string qline;
  std::string qcsv;
  bool has_qline = get_kv(a, "--query", qline);
  bool has_qcsv = get_kv(a, "--query_csv", qcsv);

  if (!has_qline && !has_qcsv) {
    std::cerr << "search: missing --query or --query_csv\n";
    return 2;
  }

  if (has_qline) {
    std::vector<float> q;
    if (!parse_query_from_string(qline, col.dim(), q, force_id)) {
      std::cerr << "search: failed to parse --query. Expect: f1,f2,...,f_dim\n";
      return 2;
    }
    auto res = filter.empty() ? col.search(q, k, ef) : col.search(q, k, ef, filter);

    std::cout << "Query=";
    print_vec(q);
    std::cout << "\nTop" << res.size() << ":\n";
    for (auto& r : res) {
      std::cout << "  index=" << r.index
                << " id=" << col.id_at(r.index)
                << " dist=" << std::fixed << std::setprecision(6) << r.distance
                << "\n";
    }
    return 0;
  }

  // query_csv (multiple queries)
  std::size_t limit = get_size_or(a, "--limit", static_cast<std::size_t>(-1));
  std::size_t count = 0;

  vecdb::csv::Options opt;
  opt.has_header = has_header;
  opt.has_id = force_id;
  opt.infer_id = !force_id;

  std::string err;
  bool ok = vecdb::csv::for_each_row(qcsv, col.dim(),
    [&](const vecdb::csv::Row& row) -> bool {
      if (limit != static_cast<std::size_t>(-1) && count >= limit) return false;

      const auto& q = row.vec;
      auto res = filter.empty() ? col.search(q, k, ef) : col.search(q, k, ef, filter);

      std::cout << "\nQuery#" << count;
      if (row.has_id) std::cout << " id=" << row.id;
      std::cout << " q=";
      print_vec(q);
      std::cout << "\nTop" << res.size() << ":\n";
      for (auto& r : res) {
        std::cout << "  index=" << r.index
                  << " id=" << col.id_at(r.index)
                  << " dist=" << std::fixed << std::setprecision(6) << r.distance
                  << "\n";
      }

      ++count;
      return true;
    }, err, opt);

  if (!ok) {
    std::cerr << "search query_csv failed: " << err << "\n";
    return 2;
  }

  return 0;
}

static int cmd_stats(const Args& a) {
  std::string dir;
  if (!get_kv(a, "--dir", dir)) { std::cerr << "stats: missing --dir\n"; return 2; }
  if (!manifest_exists(dir)) { std::cerr << "stats: collection not found (manifest.json missing): " << dir << "\n"; return 2; }

  auto col = vecdb::Collection::open(dir);

  std::cout << "Collection dir: " << col.dir() << "\n";
  std::cout << "dim: " << col.dim() << "\n";
  std::cout << "metric: " << (col.metric() == vecdb::Metric::L2 ? "l2" : "cosine") << "\n";
  std::cout << "size(slots): " << col.size() << "\n";
  std::cout << "alive: " << col.alive_count() << "\n";
  std::cout << "has_index: " << (col.has_index() ? "true" : "false") << "\n";
  return 0;
}

int main(int argc, char** argv) {
  // If no args, show help (do not auto-run heavy demos).
  if (argc <= 1) {
    print_help();
    return 0;
  }

  Args a = parse_args(argc, argv);
  if (a.pos.empty()) {
    print_help();
    return 0;
  }

  const std::string cmd = a.pos[0];

  try {
    if (cmd == "help" || cmd == "--help" || cmd == "-h") { print_help(); return 0; }
    if (cmd == "demo") return run_demo();
    if (cmd == "create") return cmd_create(a);
    if (cmd == "load") return cmd_load(a);
    if (cmd == "build") return cmd_build(a);
    if (cmd == "search") return cmd_search(a);
    if (cmd == "stats") return cmd_stats(a);

    std::cerr << "unknown command: " << cmd << "\n\n";
    print_help();
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
}
