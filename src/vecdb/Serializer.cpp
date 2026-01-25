#include "Serializer.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>
#include <cstdint>
#include <cstring>

#include "Metadata.h"

namespace vecdb {

namespace fs = std::filesystem;

static fs::path pjoin(const std::string& dir, const std::string& name) {
  return fs::path(dir) / name;
}

static void write_all_text(const fs::path& p, const std::string& s) {
  std::ofstream out(p, std::ios::binary);
  if (!out) throw std::runtime_error("Serializer: cannot open for write: " + p.string());
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
  if (!out) throw std::runtime_error("Serializer: write failed: " + p.string());
}

static std::string read_all_text(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("Serializer: cannot open for read: " + p.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static void write_u64(std::ofstream& out, std::uint64_t x) {
  out.write(reinterpret_cast<const char*>(&x), sizeof(x));
}
static std::uint64_t read_u64(std::ifstream& in) {
  std::uint64_t x = 0;
  in.read(reinterpret_cast<char*>(&x), sizeof(x));
  return x;
}

static void write_u32(std::ofstream& out, std::uint32_t x) {
  out.write(reinterpret_cast<const char*>(&x), sizeof(x));
}
static std::uint32_t read_u32(std::ifstream& in) {
  std::uint32_t x = 0;
  in.read(reinterpret_cast<char*>(&x), sizeof(x));
  return x;
}

static void write_i32(std::ofstream& out, std::int32_t x) {
  out.write(reinterpret_cast<const char*>(&x), sizeof(x));
}
static std::int32_t read_i32(std::ifstream& in) {
  std::int32_t x = 0;
  in.read(reinterpret_cast<char*>(&x), sizeof(x));
  return x;
}

static std::string metric_to_string(Metric m) {
  switch (m) {
    case Metric::L2: return "L2";
    case Metric::COSINE: return "COSINE";
    default: return "L2";
  }
}
static Metric metric_from_string(const std::string& s) {
  if (s == "L2") return Metric::L2;
  if (s == "COSINE") return Metric::COSINE;
  return Metric::L2;
}

// ---------- tiny JSON-ish parser (good enough for our fixed manifest) ----------

static std::string find_json_string(const std::string& text, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  std::size_t pos = text.find(pat);
  if (pos == std::string::npos) return "";
  pos = text.find(':', pos);
  if (pos == std::string::npos) return "";
  pos = text.find('"', pos);
  if (pos == std::string::npos) return "";
  std::size_t end = text.find('"', pos + 1);
  if (end == std::string::npos) return "";
  return text.substr(pos + 1, end - (pos + 1));
}

static std::int64_t find_json_int(const std::string& text, const std::string& key, std::int64_t def = 0) {
  std::string pat = "\"" + key + "\"";
  std::size_t pos = text.find(pat);
  if (pos == std::string::npos) return def;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return def;

  ++pos;
  while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) || text[pos] == '"')) ++pos;

  std::size_t end = pos;
  while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-')) ++end;
  if (end == pos) return def;
  return std::stoll(text.substr(pos, end - pos));
}

static double find_json_double(const std::string& text, const std::string& key, double def = 0.0) {
  std::string pat = "\"" + key + "\"";
  std::size_t pos = text.find(pat);
  if (pos == std::string::npos) return def;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return def;

  ++pos;
  while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) || text[pos] == '"')) ++pos;

  std::size_t end = pos;
  while (end < text.size() &&
         (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-' || text[end] == '.' ||
          text[end] == 'e' || text[end] == 'E' || text[end] == '+')) {
    ++end;
  }
  if (end == pos) return def;
  return std::stod(text.substr(pos, end - pos));
}

static bool find_json_bool(const std::string& text, const std::string& key, bool def = false) {
  std::string pat = "\"" + key + "\"";
  std::size_t pos = text.find(pat);
  if (pos == std::string::npos) return def;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return def;

  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;

  if (text.compare(pos, 4, "true") == 0) return true;
  if (text.compare(pos, 5, "false") == 0) return false;
  return def;
}

// ---------------- Manifest ----------------

Serializer::Manifest Serializer::read_manifest(const std::string& dir) {
  fs::path mp = pjoin(dir, "manifest.json");
  std::string text = read_all_text(mp);

  Manifest mf;
  mf.version = static_cast<int>(find_json_int(text, "version", 1));
  mf.dim = static_cast<std::size_t>(find_json_int(text, "dim", 0));
  mf.metric = metric_from_string(find_json_string(text, "metric"));

  mf.hnsw_params.M = static_cast<std::size_t>(find_json_int(text, "M", 16));
  mf.hnsw_params.M0 = static_cast<std::size_t>(find_json_int(text, "M0", 32));
  mf.hnsw_params.ef_construction = static_cast<std::size_t>(find_json_int(text, "ef_construction", 100));
  mf.hnsw_params.use_diversity = find_json_bool(text, "use_diversity", true);
  mf.hnsw_params.seed = static_cast<unsigned>(find_json_int(text, "seed", 123));
  mf.hnsw_params.level_mult = static_cast<float>(find_json_double(text, "level_mult", 1.0));

  if (mf.dim == 0) {
    throw std::runtime_error("Serializer: manifest dim invalid (0) in " + mp.string());
  }
  return mf;
}

void Serializer::write_manifest(const std::string& dir, const Manifest& mf) {
  fs::path mp = pjoin(dir, "manifest.json");

  std::ostringstream ss;
  ss << "{\n";
  ss << "  \"version\": " << mf.version << ",\n";
  ss << "  \"dim\": " << mf.dim << ",\n";
  ss << "  \"metric\": \"" << metric_to_string(mf.metric) << "\",\n";
  ss << "  \"hnsw\": {\n";
  ss << "    \"M\": " << mf.hnsw_params.M << ",\n";
  ss << "    \"M0\": " << mf.hnsw_params.M0 << ",\n";
  ss << "    \"ef_construction\": " << mf.hnsw_params.ef_construction << ",\n";
  ss << "    \"use_diversity\": " << (mf.hnsw_params.use_diversity ? "true" : "false") << ",\n";
  ss << "    \"seed\": " << mf.hnsw_params.seed << ",\n";
  ss << "    \"level_mult\": " << mf.hnsw_params.level_mult << "\n";
  ss << "  }\n";
  ss << "}\n";

  write_all_text(mp, ss.str());
}

// ---------------- VectorStore ----------------

static constexpr std::uint64_t MAGIC_VEC = 0x31565F434556uLL;   // "VECV_1" (loosely)
static constexpr std::uint64_t MAGIC_ALV = 0x31565F564C41uLL;   // "ALV_1"

void Serializer::save_store(const std::string& dir, const VectorStore& store) {
  const std::size_t N = store.size();
  const std::size_t dim = store.dim();

  // vectors.bin
  {
    fs::path vp = pjoin(dir, "vectors.bin");
    std::ofstream out(vp, std::ios::binary);
    if (!out) throw std::runtime_error("Serializer: cannot open vectors.bin for write");

    write_u64(out, MAGIC_VEC);
    write_u64(out, static_cast<std::uint64_t>(N));
    write_u64(out, static_cast<std::uint64_t>(dim));

    for (std::size_t i = 0; i < N; ++i) {
      const float* ptr = store.get_ptr(i);
      if (!ptr) {
        // dead slot -> still write stored bytes if any; our get_ptr hides dead.
        // so we write zeros; correctness doesn't depend on dead vectors.
        for (std::size_t d = 0; d < dim; ++d) {
          float z = 0.f;
          out.write(reinterpret_cast<const char*>(&z), sizeof(float));
        }
      } else {
        out.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(dim * sizeof(float)));
      }
    }

    if (!out) throw std::runtime_error("Serializer: write failed: vectors.bin");
  }

  // alive.bin
  {
    fs::path ap = pjoin(dir, "alive.bin");
    std::ofstream out(ap, std::ios::binary);
    if (!out) throw std::runtime_error("Serializer: cannot open alive.bin for write");

    write_u64(out, MAGIC_ALV);
    write_u64(out, static_cast<std::uint64_t>(N));

    for (std::size_t i = 0; i < N; ++i) {
      std::uint8_t a = store.is_alive(i) ? 1 : 0;
      out.write(reinterpret_cast<const char*>(&a), 1);
    }

    if (!out) throw std::runtime_error("Serializer: write failed: alive.bin");
  }

  // ids.txt  (v2: ALWAYS write id_at(i), even if dead, to preserve tombstones)
  {
    fs::path ip = pjoin(dir, "ids.txt");
    std::ofstream out(ip, std::ios::binary);
    if (!out) throw std::runtime_error("Serializer: cannot open ids.txt for write");

    for (std::size_t i = 0; i < N; ++i) {
      // keep stable mapping across restarts
      out << store.id_at(i) << "\n";
    }
  }

  // meta.txt
  {
    fs::path mp = pjoin(dir, "meta.txt");
    std::ofstream out(mp, std::ios::binary);
    if (!out) throw std::runtime_error("Serializer: cannot open meta.txt for write");

    for (std::size_t i = 0; i < N; ++i) {
      out << metadata::encode(store.metadata_at(i)) << "\n";
    }
  }
}

void Serializer::load_store(const std::string& dir, VectorStore& store) {
  fs::path vp = pjoin(dir, "vectors.bin");
  fs::path ap = pjoin(dir, "alive.bin");
  fs::path ip = pjoin(dir, "ids.txt");
  fs::path mp = pjoin(dir, "meta.txt");

  std::size_t N = 0;
  std::size_t dim = 0;
  std::vector<float> vectors;

  // vectors.bin
  {
    std::ifstream in(vp, std::ios::binary);
    if (!in) throw std::runtime_error("Serializer: cannot open vectors.bin for read");

    std::uint64_t magic = read_u64(in);
    if (magic != MAGIC_VEC) throw std::runtime_error("Serializer: bad vectors.bin magic");
    N = static_cast<std::size_t>(read_u64(in));
    dim = static_cast<std::size_t>(read_u64(in));

    if (dim != store.dim()) {
      throw std::runtime_error("Serializer: vectors.bin dim mismatch vs store.dim()");
    }

    vectors.resize(N * dim);
    in.read(reinterpret_cast<char*>(vectors.data()),
            static_cast<std::streamsize>(vectors.size() * sizeof(float)));
    if (!in) throw std::runtime_error("Serializer: read failed: vectors.bin");
  }

  // alive.bin
  std::vector<std::uint8_t> alive;
  {
    std::ifstream in(ap, std::ios::binary);
    if (!in) throw std::runtime_error("Serializer: cannot open alive.bin for read");

    std::uint64_t magic = read_u64(in);
    if (magic != MAGIC_ALV) throw std::runtime_error("Serializer: bad alive.bin magic");
    std::size_t n2 = static_cast<std::size_t>(read_u64(in));
    if (n2 != N) throw std::runtime_error("Serializer: alive.bin N mismatch");

    alive.resize(N);
    in.read(reinterpret_cast<char*>(alive.data()), static_cast<std::streamsize>(N));
    if (!in) throw std::runtime_error("Serializer: read failed: alive.bin");
  }

  // ids.txt
  std::vector<std::string> ids;
  {
    std::ifstream in(ip, std::ios::binary);
    if (!in) throw std::runtime_error("Serializer: cannot open ids.txt for read");

    ids.resize(N);
    std::string line;
    for (std::size_t i = 0; i < N; ++i) {
      if (!std::getline(in, line)) line.clear();
      if (!line.empty() && line.back() == '\r') line.pop_back();
      ids[i] = line;
    }
  }

  // meta.txt (optional for backward compatibility)
  std::vector<Metadata> meta;
  if (fs::exists(mp)) {
    std::ifstream in(mp, std::ios::binary);
    if (!in) throw std::runtime_error("Serializer: cannot open meta.txt for read");

    meta.resize(N);
    std::string line;
    for (std::size_t i = 0; i < N; ++i) {
      if (!std::getline(in, line)) line.clear();
      if (!line.empty() && line.back() == '\r') line.pop_back();
      std::string err;
      if (!metadata::decode(line, meta[i], err)) {
        throw std::runtime_error("Serializer: meta.txt parse error at line " +
                                 std::to_string(i + 1) + ": " + err);
      }
    }
  } else {
    meta.resize(N);
  }

  store.load_from_disk(N, vectors, alive, ids, meta);
}

// ---------------- HNSW ----------------

static const char HNSW_MAGIC[8] = {'H','N','S','W','v','1','\0','\0'};

void Serializer::save_hnsw(const std::string& dir,
                           const Hnsw& hnsw,
                           const VectorStore& store) {
  fs::path hp = pjoin(dir, "hnsw.bin");
  std::ofstream out(hp, std::ios::binary);
  if (!out) throw std::runtime_error("Serializer: cannot open hnsw.bin for write");

  out.write(HNSW_MAGIC, 8);
  write_u64(out, static_cast<std::uint64_t>(store.size()));
  write_i32(out, static_cast<std::int32_t>(hnsw.max_level()));

  Hnsw::Export ex = hnsw.export_graph();

  write_u64(out, static_cast<std::uint64_t>(ex.entry_point));
  write_u32(out, ex.has_entry ? 1u : 0u);

  for (std::size_t i = 0; i < ex.nodes.size(); ++i) {
    const auto& node = ex.nodes[i];
    write_i32(out, static_cast<std::int32_t>(node.level));
    if (node.level >= 0) {
      for (int l = 0; l <= node.level; ++l) {
        const auto& nbrs = node.links[static_cast<std::size_t>(l)];
        write_u32(out, static_cast<std::uint32_t>(nbrs.size()));
        for (std::size_t nb : nbrs) write_u32(out, static_cast<std::uint32_t>(nb));
      }
    }
  }

  if (!out) throw std::runtime_error("Serializer: write failed: hnsw.bin");
}

void Serializer::load_hnsw(const std::string& dir,
                           Hnsw& hnsw,
                           const VectorStore& store) {
  fs::path hp = pjoin(dir, "hnsw.bin");
  std::ifstream in(hp, std::ios::binary);
  if (!in) throw std::runtime_error("Serializer: cannot open hnsw.bin for read");

  char magic[8] = {0};
  in.read(magic, 8);
  if (!in || std::memcmp(magic, HNSW_MAGIC, 8) != 0) {
    throw std::runtime_error("Serializer: bad hnsw.bin magic");
  }

  std::size_t N = static_cast<std::size_t>(read_u64(in));
  if (N != store.size()) throw std::runtime_error("Serializer: hnsw.bin N mismatch vs store.size()");

  int max_level = static_cast<int>(read_i32(in));
  std::size_t entry_point = static_cast<std::size_t>(read_u64(in));
  bool has_entry = (read_u32(in) != 0);

  Hnsw::Export ex;
  ex.entry_point = entry_point;
  ex.has_entry = has_entry;
  ex.max_level = max_level;
  ex.nodes.resize(N);

  for (std::size_t i = 0; i < N; ++i) {
    int lvl = static_cast<int>(read_i32(in));
    ex.nodes[i].level = lvl;

    if (lvl >= 0) {
      ex.nodes[i].links.resize(static_cast<std::size_t>(lvl + 1));
      for (int l = 0; l <= lvl; ++l) {
        std::uint32_t deg = read_u32(in);
        auto& nbrs = ex.nodes[i].links[static_cast<std::size_t>(l)];
        nbrs.resize(deg);
        for (std::uint32_t j = 0; j < deg; ++j) {
          nbrs[j] = static_cast<std::size_t>(read_u32(in));
        }
      }
    } else {
      ex.nodes[i].links.clear();
    }
  }

  if (!in) throw std::runtime_error("Serializer: read failed: hnsw.bin");
  hnsw.import_graph(ex);
}

}  // namespace vecdb
