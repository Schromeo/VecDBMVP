#pragma once

#include <string>
#include <unordered_map>

namespace vecdb {

using Metadata = std::unordered_map<std::string, std::string>;

namespace metadata {

// Encode metadata into a single line: key=value;key2=value2 (escaped).
std::string encode(const Metadata& meta);

// Decode a metadata line into a map. Returns false on parse error.
bool decode(const std::string& line, Metadata& out, std::string& err);

}  // namespace metadata

}  // namespace vecdb
