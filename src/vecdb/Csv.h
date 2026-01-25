#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace vecdb::csv {

// A parsed CSV row that may contain an optional id column.
struct Row {
  bool has_id = false;
  std::string id;
  std::vector<float> vec;
  bool has_metadata = false;
  std::string metadata_raw;
};

// CSV parsing options.
struct Options {
  bool has_header = false;  // skip first non-empty, non-comment row
  bool has_id = false;      // force first column as id
  bool infer_id = true;     // infer id if first token is non-float
  bool allow_metadata = false; // allow a trailing metadata column
};

// Parse a single CSV line into floats (and optional id).
// Supported formats:
//  1) id,f1,f2,...,f_dim
//  2) f1,f2,...,f_dim
// Lines starting with '#' or empty lines are ignored by file readers (not here).
//
// If dim_expected > 0, enforces vec.size()==dim_expected.
// If dim_expected == 0, accepts any dim and sets vec size accordingly.
bool parse_line(const std::string& line,
                std::size_t dim_expected,
                Row& out,
                std::string& err,
                const Options& opt = Options{});

// Iterate over a CSV file (line-by-line) and parse each non-empty, non-comment line.
// On each parsed row, calls callback(row). If callback returns false, stops iteration.
// Returns true on success, false on parse or IO error (err filled).
bool for_each_row(const std::string& path,
                  std::size_t dim_expected,
                  const std::function<bool(const Row&)>& callback,
                  std::string& err,
                  const Options& opt = Options{});

}  // namespace vecdb::csv
