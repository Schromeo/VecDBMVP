cmake_minimum_required(VERSION 3.20)
project(VecDBMVP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(vecdb
  src/main.cpp
  src/vecdb/Distance.h
  src/vecdb/Distance.cpp
  src/vecdb/SearchResult.h
  src/vecdb/VectorStore.h
  src/vecdb/VectorStore.cpp
  src/vecdb/Bruteforce.h
  src/vecdb/Bruteforce.cpp
  src/vecdb/Eval.h
  src/vecdb/Eval.cpp
  src/vecdb/Hnsw0.h
  src/vecdb/Hnsw0.cpp
  src/vecdb/Hnsw.h
  src/vecdb/Hnsw.cpp
)






# Warnings (cross-platform)
if(MSVC)
  target_compile_options(vecdb PRIVATE /W4 /permissive-)
else()
  target_compile_options(vecdb PRIVATE -Wall -Wextra -Wpedantic)
endif()
