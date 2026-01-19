#pragma once

#include <cstddef>
#include <vector>

namespace vecdb {

enum class Metric {
  L2,      // squared L2 distance
  COSINE   // cosine distance = 1 - cosine_similarity
};

struct Distance {
  // squared L2 distance (no sqrt)
  static float l2_sq(const float* a, const float* b, std::size_t dim);

  // dot product
  static float dot(const float* a, const float* b, std::size_t dim);

  // L2 norm (sqrt of sum of squares)
  static float norm(const float* a, std::size_t dim);

  // in-place normalize vector to unit length (if norm is ~0, leave unchanged)
  static void normalize_inplace(float* v, std::size_t dim);

  // cosine similarity: dot(a,b)/(||a||*||b||)
  static float cosine_similarity(const float* a, const float* b, std::size_t dim);

  // cosine distance: 1 - cosine_similarity
  static float cosine_distance(const float* a, const float* b, std::size_t dim);

  // Unified distance API (lower is closer)
  static float distance(Metric metric, const float* a, const float* b, std::size_t dim);
};

} // namespace vecdb
