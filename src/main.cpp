#include <iostream>
#include <vector>
#include <iomanip>

#include "vecdb/Distance.h"

static void print_vec(const std::vector<float>& v) {
  std::cout << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i];
    if (i + 1 != v.size()) std::cout << ", ";
  }
  std::cout << "]";
}

int main() {
  std::cout << "VecDB MVP starting..." << std::endl;

#ifdef _WIN32
  std::cout << "Platform: Windows" << std::endl;
#elif __APPLE__
  std::cout << "Platform: macOS" << std::endl;
#elif __linux__
  std::cout << "Platform: Linux" << std::endl;
#else
  std::cout << "Platform: Unknown" << std::endl;
#endif

  using vecdb::Distance;
  using vecdb::Metric;

  // Quick sanity tests
  std::vector<float> a{1.0f, 0.0f};
  std::vector<float> b{2.0f, 0.0f};
  std::vector<float> c{0.0f, 1.0f};

  std::cout << std::fixed << std::setprecision(6);

  std::cout << "\nDistance sanity checks:\n";
  std::cout << "a="; print_vec(a); std::cout << "  b="; print_vec(b); std::cout << "  c="; print_vec(c); std::cout << "\n";

  float l2_ab = Distance::distance(Metric::L2, a.data(), b.data(), a.size());
  float l2_ac = Distance::distance(Metric::L2, a.data(), c.data(), a.size());
  std::cout << "L2^2(a,b) = " << l2_ab << "  (expected 1)\n";
  std::cout << "L2^2(a,c) = " << l2_ac << "  (expected 2)\n";

  float cos_ab = Distance::distance(Metric::COSINE, a.data(), b.data(), a.size());
  float cos_ac = Distance::distance(Metric::COSINE, a.data(), c.data(), a.size());
  std::cout << "cosDist(a,b) = " << cos_ab << "  (expected 0, same direction)\n";
  std::cout << "cosDist(a,c) = " << cos_ac << "  (expected 1, orthogonal)\n";

  std::vector<float> d{3.0f, 4.0f}; // norm 5
  Distance::normalize_inplace(d.data(), d.size());
  std::cout << "normalize([3,4]) = "; print_vec(d); std::cout << "  (expected [0.6,0.8])\n";

  std::cout << "\nNext: implement VectorStore (contiguous storage + id->index)\n";
  return 0;
}
