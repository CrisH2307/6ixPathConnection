import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "people.json"

with DATA_PATH.open("r", encoding="utf-8") as f:
    people = json.load(f)

# collect all buckets that appear at least once
all_buckets = sorted({b for p in people for b in p.get("skills", [])})

def person_features(p):
    vec = []
    cats = set(p.get("skills", []))
    for b in all_buckets:
        vec.append(1.0 if b in cats else 0.0)
    return np.array(vec, dtype="float32")

X = np.vstack([person_features(p) for p in people])  # shape: (n_people, n_buckets)
ids = [str(p["_id"]) for p in people]
names = [p["name"] for p in people]
print("Feature matrix shape:", X.shape)

# Perform KMeans clustering
k = 4
kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
labels = kmeans.fit_predict(X)   # array of cluster ids (0..k-1)
print("Cluster labels:", len(labels), labels)


# Inspect clusters
df = pd.DataFrame(X, columns=all_buckets)
df["skills"] = labels
df["name"] = names
df["_id"] = ids

# cluster profile = mean of features in cluster (which buckets are common)
cluster_profiles = df.groupby("skills")[all_buckets].mean().round(2)
print(cluster_profiles.sort_index())

# list members per cluster
for c in range(k):
    members = df[df["skills"] == c][["name","_id"]].values.tolist()
    print(f"\nCluster {c} ({len(members)} people):")
    for name, pid in members[:10]:
        print(" -", name, f"({pid})")


# 2D PCA plot of clusters
# pca = PCA(n_components=2, random_state=42)
# XY = pca.fit_transform(X)

# plt.figure(figsize=(6,5))
# for c in range(k):
#     mask = labels == c
#     plt.scatter(XY[mask,0], XY[mask,1], label=f"Cluster {c}", s=40)
# plt.legend()
# plt.title("People clusters (PCA 2D)")
# plt.xlabel("PC1"); plt.ylabel("PC2")
# plt.tight_layout()
# plt.show()

# Save cluster assignments
cluster_map = {pid: int(lbl) for pid, lbl in zip(ids, labels)}
for p in people:
    p["skills"] = cluster_map[str(p["_id"])]

print(cluster_map)

