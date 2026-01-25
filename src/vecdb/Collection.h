#pragma once

#include <cstddef>
#include <memory>
#include <shared_mutex>
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

  Collection(const Collection&) = delete;
  Collection& operator=(const Collection&) = delete;
  Collection(Collection&& other) noexcept;
  Collection& operator=(Collection&& other) noexcept;

  std::size_t dim() const;
  Metric metric() const;
  const std::string& dir() const;

  // slots (includes dead)
  std::size_t size() const;

  // alive count (computed)
  std::size_t alive_count() const;

  // printing / debug helper
  const std::string& id_at(std::size_t index) const;

  // metadata helper
  const Metadata& metadata_at(std::size_t index) const;
  const Metadata* metadata_of(const std::string& id) const;

  // --- mutation ---
  std::size_t upsert(const std::string& id, const std::vector<float>& vec);
  std::size_t upsert(const std::string& id, const std::vector<float>& vec, const Metadata& meta);
  bool remove(const std::string& id);
  bool contains(const std::string& id) const;

  // --- index ---
  void build_index();
  bool has_index() const;

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
  mutable std::shared_mutex mtx_;
};

}  // namespace vecdb
