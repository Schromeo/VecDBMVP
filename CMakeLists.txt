cmake_minimum_required(VERSION 3.20)
project(VecDBMVP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Reasonable warnings (works for GCC/Clang/MinGW)
if (MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Collect sources automatically (so adding new files doesn't require editing CMake)
file(GLOB_RECURSE VECDB_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(vecdb ${VECDB_SOURCES})

target_include_directories(vecdb PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
)

# On Windows, filesystem is standard in C++17; no extra link needed for modern toolchains.
# If you ever hit older GCC issues, you might need: target_link_libraries(vecdb stdc++fs)
