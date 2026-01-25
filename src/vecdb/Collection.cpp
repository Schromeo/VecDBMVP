#include "Collection.h"

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>

#include "Serializer.h"

namespace vecdb {

namespace fs = std::filesystem;

static void ensure_dir_exists(const std::string& dir) {
  fs::path p(dir);
  if (fs::exists(p)) {
    if (!fs::is_directory(p)) {
      throw std::runtime_error("Collection: path exists but is not a directory: " + dir);
    }
  } else {
    fs::create_directories(p);
  }
}

static bool file_exists(const fs::path& p) {
  std::error_code ec;
  return fs::exists(p, ec) && fs::is_regular_file(p, ec);
}

Collection::Collection(std::string dir, Options opt)
    : dir_(std::move(dir)),
      opt_(opt),
      store_(opt_.dim),
      hnsw_(nullptr) {
  if (opt_.dim == 0) throw std::invalid_argument("Collection: dim must be > 0");
}

Collection::Collection(Collection&& other) noexcept
    : dir_(std::move(other.dir_)),
      opt_(other.opt_),
      store_(std::move(other.store_)),
      hnsw_(std::move(other.hnsw_)) {}

Collection& Collection::operator=(Collection&& other) noexcept {
  if (this == &other) return *this;
  std::scoped_lock lock(mtx_, other.mtx_);
  dir_ = std::move(other.dir_);
  opt_ = other.opt_;
  store_ = std::move(other.store_);
  hnsw_ = std::move(other.hnsw_);
  return *this;
}

Collection Collection::create(const std::string& dir, Options opt) {
  ensure_dir_exists(dir);
  Collection c(dir, opt);
  c.save();
  return c;
}

Collection Collection::open(const std::string& dir) {
  ensure_dir_exists(dir);

  Serializer::Manifest mf = Serializer::read_manifest(dir);

  Options opt;
  opt.dim = mf.dim;
  opt.metric = mf.metric;
  opt.hnsw_params = mf.hnsw_params;

  Collection c(dir, opt);
  c.load();
  return c;
}

std::size_t Collection::dim() const {
  std::shared_lock lock(mtx_);
  return opt_.dim;
}

Metric Collection::metric() const {
  std::shared_lock lock(mtx_);
  return opt_.metric;
}

const std::string& Collection::dir() const {
  std::shared_lock lock(mtx_);
  return dir_;
}

std::size_t Collection::size() const {
  std::shared_lock lock(mtx_);
  return store_.size();
}

std::size_t Collection::alive_count() const {
  std::shared_lock lock(mtx_);
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < store_.size(); ++i) {
    if (store_.is_alive(i)) ++cnt;
  }
  return cnt;
}

const std::string& Collection::id_at(std::size_t index) const {
  std::shared_lock lock(mtx_);
  return store_.id_at(index);
}

const Metadata& Collection::metadata_at(std::size_t index) const {
  std::shared_lock lock(mtx_);
  return store_.metadata_at(index);
}

const Metadata* Collection::metadata_of(const std::string& id) const {
  std::shared_lock lock(mtx_);
  return store_.metadata_ptr(id);
}

void Collection::set_metric(Metric m) {
  std::unique_lock lock(mtx_);
  opt_.metric = m;
  if (hnsw_) hnsw_.reset();
}

void Collection::set_hnsw_params(Hnsw::Params p) {
  std::unique_lock lock(mtx_);
  opt_.hnsw_params = p;
  if (hnsw_) hnsw_.reset();
}

std::size_t Collection::upsert(const std::string& id, const std::vector<float>& vec) {
  return upsert(id, vec, Metadata{});
}

std::size_t Collection::upsert(const std::string& id,
                               const std::vector<float>& vec,
                               const Metadata& meta) {
  std::unique_lock lock(mtx_);
  if (vec.size() != opt_.dim) throw std::invalid_argument("Collection::upsert: vector dim mismatch");

  std::size_t idx = store_.upsert(id, vec, meta);

  // v1 correctness-first: any mutation invalidates index (rebuild later).
  if (hnsw_) hnsw_.reset();
  return idx;
}

bool Collection::remove(const std::string& id) {
  std::unique_lock lock(mtx_);
  bool ok = store_.remove(id);
  if (ok && hnsw_) hnsw_.reset();
  return ok;
}

bool Collection::contains(const std::string& id) const {
  std::shared_lock lock(mtx_);
  return store_.contains(id);
}

bool Collection::has_index() const {
  std::shared_lock lock(mtx_);
  return hnsw_ != nullptr;
}

void Collection::build_index() {
  std::unique_lock lock(mtx_);
  hnsw_ = std::make_unique<Hnsw>(store_, opt_.metric, opt_.hnsw_params);
  for (std::size_t i = 0; i < store_.size(); ++i) {
    if (store_.is_alive(i)) hnsw_->insert(i);
  }
}

void Collection::ensure_index_ready() const {
  if (!hnsw_) {
    throw std::runtime_error(
        "Collection: index not ready. Call build_index() or open() a collection with an index saved.");
  }
}

std::vector<SearchResult> Collection::search(const std::vector<float>& query,
                                             std::size_t k,
                                             std::size_t ef_search) const {
  std::shared_lock lock(mtx_);
  if (query.size() != opt_.dim) throw std::invalid_argument("Collection::search: query dim mismatch");
  ensure_index_ready();
  return hnsw_->search(query, k, ef_search);
}

static bool metadata_matches(const Metadata& meta, const Collection::MetadataFilter& filter) {
  if (filter.empty()) return true;
  auto it = meta.find(filter.key);
  return it != meta.end() && it->second == filter.value;
}

std::vector<SearchResult> Collection::search(const std::vector<float>& query,
                                             std::size_t k,
                                             std::size_t ef_search,
                                             const MetadataFilter& filter) const {
  std::shared_lock lock(mtx_);
  if (query.size() != opt_.dim) throw std::invalid_argument("Collection::search: query dim mismatch");

  if (filter.empty()) {
    ensure_index_ready();
    return hnsw_->search(query, k, ef_search);
  }

  // Filtered search (exact scan for correctness). Can be optimized later.
  std::vector<SearchResult> heap;
  heap.reserve(k + 1);

  for (std::size_t i = 0; i < store_.size(); ++i) {
    if (!store_.is_alive(i)) continue;
    if (!metadata_matches(store_.metadata_at(i), filter)) continue;

    const float* p = store_.get_ptr(i);
    if (!p) continue;

    float d = Distance::distance(opt_.metric, query.data(), p, opt_.dim);

    if (heap.size() < k) {
      heap.push_back({i, d});
      if (heap.size() == k) {
        std::make_heap(heap.begin(), heap.end(),
                       [](const auto& a, const auto& b) { return a.distance < b.distance; });
      }
    } else if (d < heap.front().distance) {
      std::pop_heap(heap.begin(), heap.end(),
                    [](const auto& a, const auto& b) { return a.distance < b.distance; });
      heap.back() = {i, d};
      std::push_heap(heap.begin(), heap.end(),
                     [](const auto& a, const auto& b) { return a.distance < b.distance; });
    }
  }

  if (heap.empty()) return heap;
  std::sort(heap.begin(), heap.end(),
            [](const auto& a, const auto& b) { return a.distance < b.distance; });
  return heap;
}

void Collection::save() const {
  std::unique_lock lock(mtx_);
  ensure_dir_exists(dir_);

  Serializer::Manifest mf;
  mf.version = 1;
  mf.dim = opt_.dim;
  mf.metric = opt_.metric;
  mf.hnsw_params = opt_.hnsw_params;

  Serializer::write_manifest(dir_, mf);
  Serializer::save_store(dir_, store_);

  if (hnsw_) {
    Serializer::save_hnsw(dir_, *hnsw_, store_);
  } else {
    fs::path hnsw_path = fs::path(dir_) / "hnsw.bin";
    std::error_code ec;
    if (file_exists(hnsw_path)) fs::remove(hnsw_path, ec);
  }
}

void Collection::load() {
  std::unique_lock lock(mtx_);
  Serializer::load_store(dir_, store_);

  fs::path hnsw_path = fs::path(dir_) / "hnsw.bin";
  if (file_exists(hnsw_path)) {
    hnsw_ = std::make_unique<Hnsw>(store_, opt_.metric, opt_.hnsw_params);
    Serializer::load_hnsw(dir_, *hnsw_, store_);
  } else {
    hnsw_.reset();
  }
}

}  // namespace vecdb
