#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "Distance.h"
#include "Metadata.h"
#include "SearchResult.h"
#include "VectorStore.h"
#include "Hnsw.h"

namespace vecdb {

class Collection {
 public:
  struct Options {
    std::size_t dim = 0;
    Metric metric = Metric::L2;
    Hnsw::Params hnsw_params{};
  };

  static Collection create(const std::string& dir, Options opt);
  static Collection open(const std::string& dir);

  std::size_t dim() const { return opt_.dim; }
  Metric metric() const { return opt_.metric; }
  const std::string& dir() const { return dir_; }

  // slots (includes dead)
  std::size_t size() const { return store_.size(); }

  // alive count (computed)
  std::size_t alive_count() const;

  // printing / debug helper
  const std::string& id_at(std::size_t index) const { return store_.id_at(index); }

  // metadata helper
  const Metadata& metadata_at(std::size_t index) const { return store_.metadata_at(index); }
  const Metadata* metadata_of(const std::string& id) const { return store_.metadata_ptr(id); }

  // --- mutation ---
  std::size_t upsert(const std::string& id, const std::vector<float>& vec);
  std::size_t upsert(const std::string& id, const std::vector<float>& vec, const Metadata& meta);
  bool remove(const std::string& id);
  bool contains(const std::string& id) const { return store_.contains(id); }

  // --- index ---
  void build_index();
  bool has_index() const { return hnsw_ != nullptr; }

  // Allow CLI to override index parameters before build_index()
  void set_metric(Metric m);
  void set_hnsw_params(Hnsw::Params p);

  struct MetadataFilter {
    std::string key;
    std::string value;
    bool empty() const { return key.empty(); }
  };

  std::vector<SearchResult> search(const std::vector<float>& query,
                                  std::size_t k,
                                  std::size_t ef_search) const;

  std::vector<SearchResult> search(const std::vector<float>& query,
                                   std::size_t k,
                                   std::size_t ef_search,
                                   const MetadataFilter& filter) const;

  // --- persistence ---
  void save() const;
  void load();

 private:
  Collection(std::string dir, Options opt);
  void ensure_index_ready() const;

  std::string dir_;
  Options opt_;
  VectorStore store_;
  std::unique_ptr<Hnsw> hnsw_;
};

}  // namespace vecdb
