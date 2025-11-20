
# K-Means From Scratch — Starter

This is a tiny scaffold to help you implement K-Means yourself.

## What to do

1. Open `kmeans_scratch.py` and read the TODOs.
2. Implement the missing pieces (they're mostly filled, but try rewriting them yourself if you want!).
3. Run tests:
   ```bash
   python -m pytest -q
   ```

## Suggested implementation order
- `_pairwise_distances`
- `_init_centroids` (random first; k-means++ optional)
- `_assign_clusters`
- `_update_centroids`
- `fit` loop (stop when centroid shift < `tol` or `max_iters` reached)
- `predict`
- `inertia_`

## Stretch goals
- Implement **k-means++** initialization.
- Add `score` method that returns `-inertia_` like scikit-learn.
- Add a method to track inertia per iteration for learning curves.
- Standardize features before clustering (e.g., with z-scores).
