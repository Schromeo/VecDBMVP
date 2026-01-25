#include "Metadata.h"

#include <algorithm>
#include <sstream>
#include <vector>

namespace vecdb::metadata {

static std::string escape_token(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\\' || c == ';' || c == '=') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

static bool unescape_token(const std::string& s, std::string& out) {
  out.clear();
  out.reserve(s.size());
  bool esc = false;
  for (char c : s) {
    if (esc) {
      out.push_back(c);
      esc = false;
    } else if (c == '\\') {
      esc = true;
    } else {
      out.push_back(c);
    }
  }
  return !esc; // dangling escape is invalid
}

std::string encode(const Metadata& meta) {
  if (meta.empty()) return std::string();

  std::vector<std::pair<std::string, std::string>> items(meta.begin(), meta.end());
  std::sort(items.begin(), items.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  std::ostringstream ss;
  bool first = true;
  for (const auto& kv : items) {
    if (!first) ss << ';';
    first = false;
    ss << escape_token(kv.first) << '=' << escape_token(kv.second);
  }
  return ss.str();
}

bool decode(const std::string& line, Metadata& out, std::string& err) {
  out.clear();
  if (line.empty()) return true;

  std::string key_raw;
  std::string val_raw;
  std::string token;
  bool in_key = true;
  bool esc = false;

  auto flush_pair = [&]() -> bool {
    std::string key;
    std::string val;
    if (!unescape_token(key_raw, key) || !unescape_token(val_raw, val)) {
      err = "metadata escape error";
      return false;
    }
    if (!key.empty()) out[key] = val;
    key_raw.clear();
    val_raw.clear();
    return true;
  };

  for (char c : line) {
    if (esc) {
      (in_key ? key_raw : val_raw).push_back(c);
      esc = false;
      continue;
    }
    if (c == '\\') {
      esc = true;
      continue;
    }
    if (in_key && c == '=') {
      in_key = false;
      continue;
    }
    if (!in_key && c == ';') {
      if (!flush_pair()) return false;
      in_key = true;
      continue;
    }
    (in_key ? key_raw : val_raw).push_back(c);
  }

  if (esc) {
    err = "metadata trailing escape";
    return false;
  }
  if (!key_raw.empty() || !val_raw.empty()) {
    if (!flush_pair()) return false;
  }
  return true;
}

}  // namespace vecdb::metadata
