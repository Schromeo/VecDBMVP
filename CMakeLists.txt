cmake_minimum_required(VERSION 3.20)
project(VecDBMVP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings
if (MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------- Core library (all vecdb implementation files) ----------------
file(GLOB_RECURSE VECDB_CORE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/vecdb/*.cpp"
)

add_library(vecdb_core ${VECDB_CORE_SOURCES})
target_include_directories(vecdb_core PUBLIC "${CMAKE_SOURCE_DIR}/src")

# ---------------- Demo executable (your current main) ----------------
add_executable(vecdb "${CMAKE_SOURCE_DIR}/src/main.cpp")
target_link_libraries(vecdb PRIVATE vecdb_core)

# ---------------- Tests executable ----------------
enable_testing()

add_executable(vecdb_tests "${CMAKE_SOURCE_DIR}/tests/TestMain.cpp")
target_link_libraries(vecdb_tests PRIVATE vecdb_core)

add_test(NAME vecdb_tests COMMAND vecdb_tests)
