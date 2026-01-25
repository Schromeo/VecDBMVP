#include "Collection.h"

#include <filesystem>
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

std::size_t Collection::upsert(const std::string& id, const std::vector<float>& vec) {
  if (vec.size() != opt_.dim) throw std::invalid_argument("Collection::upsert: vector dim mismatch");

  std::size_t idx = store_.upsert(id, vec);

  // v1 correctness-first: any mutation invalidates index (rebuild later).
  if (hnsw_) hnsw_.reset();
  return idx;
}

bool Collection::remove(const std::string& id) {
  bool ok = store_.remove(id);
  if (ok && hnsw_) hnsw_.reset();
  return ok;
}

void Collection::build_index() {
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
  if (query.size() != opt_.dim) throw std::invalid_argument("Collection::search: query dim mismatch");
  ensure_index_ready();
  return hnsw_->search(query, k, ef_search);
}

void Collection::save() const {
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
