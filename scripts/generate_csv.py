import csv
import random
from pathlib import Path

random.seed(123)

# Output paths
root = Path(__file__).resolve().parents[1]
out_dir = root / "data"
out_dir.mkdir(parents=True, exist_ok=True)

vectors_csv = out_dir / "vectors.csv"
vectors_numeric_csv = out_dir / "vectors_numeric_id.csv"
vectors_meta_csv = out_dir / "vectors_with_meta.csv"
queries_csv = out_dir / "queries.csv"
queries_with_id_csv = out_dir / "queries_with_id.csv"

# Config (more realistic)
num_vectors = 1000
num_queries = 50
dim = 16
num_clusters = 5
cluster_std = 0.08

def make_cluster_centers(k: int, d: int):
    return [[random.uniform(-1, 1) for _ in range(d)] for _ in range(k)]

def sample_vec(center):
    return [random.gauss(mu, cluster_std) for mu in center]

centers = make_cluster_centers(num_clusters, dim)

# Generate vectors with ids (string id)
with vectors_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id"] + [f"v{i}" for i in range(dim)])
    for i in range(num_vectors):
        c = centers[i % num_clusters]
        vec = sample_vec(c)
        row = [f"item-{i:06d}"] + [f"{v:.6f}" for v in vec]
        w.writerow(row)

# Generate vectors with numeric id (for --has-id testing)
with vectors_numeric_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id"] + [f"v{i}" for i in range(dim)])
    for i in range(num_vectors):
        c = centers[i % num_clusters]
        vec = sample_vec(c)
        row = [str(i)] + [f"{v:.6f}" for v in vec]
        w.writerow(row)

# Generate vectors with metadata (key=value;key2=value2)
with vectors_meta_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id"] + [f"v{i}" for i in range(dim)] + ["meta"])
    for i in range(num_vectors):
        c = centers[i % num_clusters]
        vec = sample_vec(c)
        meta = f"cluster={i % num_clusters};source=synthetic"
        row = [f"item-{i:06d}"] + [f"{v:.6f}" for v in vec] + [meta]
        w.writerow(row)

# Generate queries (no id)
with queries_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"v{i}" for i in range(dim)])
    for i in range(num_queries):
        c = centers[i % num_clusters]
        vec = sample_vec(c)
        row = [f"{v:.6f}" for v in vec]
        w.writerow(row)

# Generate queries with id (string id)
with queries_with_id_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id"] + [f"v{i}" for i in range(dim)])
    for i in range(num_queries):
        c = centers[i % num_clusters]
        vec = sample_vec(c)
        row = [f"q-{i:04d}"] + [f"{v:.6f}" for v in vec]
        w.writerow(row)

print(f"Wrote {vectors_csv}")
print(f"Wrote {vectors_numeric_csv}")
print(f"Wrote {vectors_meta_csv}")
print(f"Wrote {queries_csv}")
print(f"Wrote {queries_with_id_csv}")
print(
    f"dim={dim}, vectors={num_vectors}, queries={num_queries}, "
    f"clusters={num_clusters}, cluster_std={cluster_std}"
)
