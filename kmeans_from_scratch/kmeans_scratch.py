"""K-Means from scratch (scaffold)
===================================
Fill in the TODOs to implement a working K-Means algorithm.

Recommended order (TDD-style):
1) implement `_pairwise_distances`
2) implement `_init_centroids` (random init first; k-means++ is optional stretch)
3) implement `_assign_clusters`
4) implement `_update_centroids`
5) implement `fit` loop (convergence by centroid shift < tol)
6) implement `predict`
7) compute `inertia_` (sum of squared distances to assigned centroids)

Run tests: `python -m pytest -q` inside the project root.

Author: Todd Glad ✨
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class KMeansScratch:
    n_clusters: int
    max_iters: int = 300
    tol: float = 1e-4
    init: str = "random"  # {"random", "k-means++" (optional)}
    random_state: Optional[int] = None
    verbose: bool = False

    # Learned attributes (populated after fit)
    centroids_: Optional[np.ndarray] = None
    labels_: Optional[np.ndarray] = None
    n_iter_: Optional[int] = None
    inertia_: Optional[float] = None

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    # ---------- Core building blocks ----------
    @staticmethod
    def _pairwise_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances between each row in X and each centroid in C.
        Shape: X (n_samples, n_features), C (k, n_features) -> (n_samples, k)
        TODO: Implement efficiently with vectorized numpy (no loops).
        Hint: use (a-b)^2 = a^2 + b^2 - 2ab
        """
        # BEGIN TODO
        # raise NotImplementedError
        X2 = np.sum(X**2, axis=1, keepdims=True)  # (n_samples, 1)
        C2 = np.sum(C**2, axis=1, keepdims=True).T  # (1, k)
        XC = X @ C.T  # (n_samples, k)
        d2 = X2 + C2 - 2 * XC  # (n_samples
        return np.maximum(d2, 0.0)  # Ensure non-negative distances

        # END TODO

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids.
        Start with 'random' (choose k distinct points from X).
        Optional stretch: implement k-means++ (set self.init == 'k-means++').
        """
        rng = self._rng()
        n = X.shape[0]
        if self.n_clusters > n:
            raise ValueError("n_clusters cannot exceed number of samples")
        if self.init == "random":
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].astype(float, copy=True)
        elif self.init == "k-means++":
            # OPTIONAL: Implement k-means++ as a stretch goal
            # For now, fall back to random
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].astype(float, copy=True)
        else:
            raise ValueError(f"Unknown init: {self.init}")

    @staticmethod
    def _assign_clusters(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Return cluster index (0..k-1) for each row of X, given centroids C.
        TODO: use _pairwise_distances to compute nearest centroid.
        """
        # BEGIN TODO
        d2 = KMeansScratch._pairwise_distances(X, C)
        return np.argmin(d2, axis=1)
        # END TODO

    @staticmethod
    def _update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        """Compute new centroids as mean of assigned points.
        Handle empty clusters by re-seeding that centroid to a random data point.
        """
        C = np.zeros((k, X.shape[1]), dtype=float)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                C[j] = X[mask].mean(axis=0)
            else:
                # Re-seed empty cluster to a random point
                idx = np.random.randint(0, X.shape[0])
                C[j] = X[idx]
        return C

    @staticmethod
    def _compute_inertia(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
        """Sum of squared distances of samples to their assigned centroid."""
        d2 = KMeansScratch._pairwise_distances(X, C)
        rows = np.arange(X.shape[0])
        return float(np.sum(d2[rows, labels]))

    # ---------- Public API ----------
    def fit(self, X: np.ndarray) -> "KMeansScratch":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        rng = self._rng()

        # Step 1: init centroids
        C = self._init_centroids(X)

        last_shift = np.inf
        for it in range(1, self.max_iters + 1):
            # Step 2: assign
            labels = self._assign_clusters(X, C)

            # Step 3: update
            new_C = self._update_centroids(X, labels, self.n_clusters)

            # Step 4: check convergence
            shift = float(np.linalg.norm(new_C - C))
            if self.verbose:
                print(f"iter={it:03d} shift={shift:.6f}")
            if shift <= self.tol:
                C = new_C
                self.n_iter_ = it
                break

            C = new_C
            last_shift = shift
        else:
            # max iters reached
            self.n_iter_ = self.max_iters

        self.centroids_ = C
        self.labels_ = self._assign_clusters(X, C)
        self.inertia_ = self._compute_inertia(X, C, self.labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X = np.asarray(X, dtype=float)
        return self._assign_clusters(X, self.centroids_)
