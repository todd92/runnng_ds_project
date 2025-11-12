# test_run.py
import numpy as np
from kmeans_more_testing import KMeans

# --- 1. Generate simple dummy data ---
# Let's make 6 points in 2D space.
# It's obvious to us there are 2 clusters here: bottom-left and top-right.
X = np.array(
    [
        [1, 2],
        [1.5, 1.8],
        [1, 0.6],  # Cluster 1 (small numbers)
        [10, 8],
        [8, 8],
        [9, 11],  # Cluster 2 (big numbers)
    ]
)

print("Data shape:", X.shape)
# Expected output: (6, 2)  -> 6 rows (points), 2 columns (dimensions/features)

# --- 2. Instantiate your model ---
# We know there are 2 clusters, so let's set k=2
model = KMeans(k=2, max_iters=3)

# --- 3. Attempt to run it ---
print("\nStarting fit()...")
model.fit(X)
print("Fit() finished.")

# --- 4. Check results ---
# Right now, these will be None because we haven't implemented the logic yet.
print("\nFinal Centroids:\n", model.centroids)
print("Final Labels:\n", model.labels)
