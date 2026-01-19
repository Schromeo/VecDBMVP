#include <iostream>

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

  std::cout << "Next: implement Distance + VectorStore + HNSW" << std::endl;
  return 0;
}
