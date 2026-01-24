#pragma once

#include <cstddef>

namespace vecdb {

struct SearchResult {
  std::size_t index;  // internal index in VectorStore
  float distance;     // lower is closer
};

}  // namespace vecdb
