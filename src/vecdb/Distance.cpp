#include "Distance.h"

#include <cmath>
#include <algorithm>

namespace vecdb {

float Distance::l2_sq(const float* a, const float* b, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float Distance::dot(const float* a, const float* b, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float Distance::norm(const float* a, std::size_t dim) {
  return std::sqrt(dot(a, a, dim));
}

void Distance::normalize_inplace(float* v, std::size_t dim) {
  float n = norm(v, dim);
  // avoid divide-by-zero; also avoid blowing up for extremely tiny norm
  if (n < 1e-12f) return;
  float inv = 1.0f / n;
  for (std::size_t i = 0; i < dim; ++i) {
    v[i] *= inv;
  }
}

float Distance::cosine_similarity(const float* a, const float* b, std::size_t dim) {
  float denom = norm(a, dim) * norm(b, dim);
  if (denom < 1e-12f) return 0.0f;
  return dot(a, b, dim) / denom;
}

float Distance::cosine_distance(const float* a, const float* b, std::size_t dim) {
  // cosine distance in [0,2] typically (if similarity in [-1,1])
  return 1.0f - cosine_similarity(a, b, dim);
}

float Distance::distance(Metric metric, const float* a, const float* b, std::size_t dim) {
  switch (metric) {
    case Metric::L2:
      return l2_sq(a, b, dim);
    case Metric::COSINE:
      return cosine_distance(a, b, dim);
    default:
      // fallback
      return l2_sq(a, b, dim);
  }
}

} // namespace vecdb
