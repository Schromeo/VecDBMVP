#include "Csv.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace vecdb::csv {

static inline void trim_inplace(std::string& s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

static std::vector<std::string> split_csv_quoted(const std::string& line) {
  // Basic CSV split with quoted fields (RFC4180-ish).
  std::vector<std::string> parts;
  std::string cur;
  bool in_quotes = false;

  for (std::size_t i = 0; i < line.size(); ++i) {
    char ch = line[i];
    if (in_quotes) {
      if (ch == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          cur.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        cur.push_back(ch);
      }
    } else {
      if (ch == '"') {
        in_quotes = true;
      } else if (ch == ',') {
        parts.push_back(cur);
        cur.clear();
      } else {
        cur.push_back(ch);
      }
    }
  }
  parts.push_back(cur);
  for (auto& p : parts) trim_inplace(p);
  return parts;
}

static bool parse_float(const std::string& s, float& out) {
  // robust-ish parse using strtof
  char* end = nullptr;
  errno = 0;
  const float v = std::strtof(s.c_str(), &end);
  if (end == s.c_str()) return false;      // no conversion
  if (errno == ERANGE) return false;        // out of range
  // allow trailing spaces
  while (*end) {
    if (!std::isspace(static_cast<unsigned char>(*end))) return false;
    ++end;
  }
  out = v;
  return true;
}

bool parse_line(const std::string& line,
                std::size_t dim_expected,
                Row& out,
                std::string& err,
                const Options& opt) {
  out = Row{};

  auto parts = split_csv_quoted(line);
  if (parts.empty()) {
    err = "empty csv line";
    return false;
  }

  std::size_t start = 0;
  if (opt.has_id) {
    out.has_id = true;
    out.id = parts[0];
    start = 1;
  } else if (opt.infer_id) {
    float tmp = 0.0f;
    const bool first_is_float = parse_float(parts[0], tmp);
    if (!first_is_float) {
      out.has_id = true;
      out.id = parts[0];
      start = 1;
    }
  }

  if (start >= parts.size()) {
    err = "no vector values found";
    return false;
  }

  std::size_t remaining = parts.size() - start;
  bool has_meta = false;
  if (opt.allow_metadata && dim_expected > 0) {
    if (remaining == dim_expected + 1) {
      has_meta = true;
    } else if (remaining > dim_expected + 1) {
      err = "too many columns (metadata expects exactly one extra column)";
      return false;
    }
  }

  std::size_t vec_count = remaining - (has_meta ? 1 : 0);
  if (dim_expected > 0 && vec_count != dim_expected) {
    err = "dimension mismatch: expected dim=" + std::to_string(dim_expected) +
          " got dim=" + std::to_string(vec_count);
    return false;
  }

  out.vec.clear();
  out.vec.reserve(vec_count);
  for (std::size_t i = start; i < start + vec_count; ++i) {
    float v = 0.0f;
    if (!parse_float(parts[i], v)) {
      err = "failed to parse float at column " + std::to_string(i + 1) + ": '" + parts[i] + "'";
      return false;
    }
    out.vec.push_back(v);
  }

  if (has_meta) {
    out.has_metadata = true;
    out.metadata_raw = parts[start + vec_count];
  }

  return true;
}

bool for_each_row(const std::string& path,
                  std::size_t dim_expected,
                  const std::function<bool(const Row&)>& callback,
                  std::string& err,
                  const Options& opt) {
  std::ifstream in(path);
  if (!in) {
    err = "failed to open file: " + path;
    return false;
  }

  std::string line;
  std::size_t line_no = 0;
  bool header_skipped = false;
  while (std::getline(in, line)) {
    ++line_no;
    std::string t = line;
    if (line_no == 1 && t.size() >= 3 &&
        static_cast<unsigned char>(t[0]) == 0xEF &&
        static_cast<unsigned char>(t[1]) == 0xBB &&
        static_cast<unsigned char>(t[2]) == 0xBF) {
      t = t.substr(3);
    }
    trim_inplace(t);
    if (t.empty()) continue;
    if (!t.empty() && t[0] == '#') continue;

    if (opt.has_header && !header_skipped) {
      header_skipped = true;
      continue;
    }

    Row row;
    std::string perr;
    if (!parse_line(t, dim_expected, row, perr, opt)) {
      err = "CSV parse error at line " + std::to_string(line_no) + ": " + perr;
      return false;
    }

    if (!callback(row)) return true;
  }

  return true;
}

}  // namespace vecdb::csv
