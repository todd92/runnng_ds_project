import numpy as np
import pytest

from kmeans_scratch import KMeansScratch

print("This is a test")


def make_simple_blobs(n=300, centers=3, seed=42):
    rng = np.random.default_rng(seed)
    C = rng.uniform(-5, 5, size=(centers, 2))
    X = np.vstack(
        [
            rng.normal(loc=C[j], scale=0.4, size=(n // centers, 2))
            for j in range(centers)
        ]
    )
    return X, C


def test_pairwise_distances():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    C = np.array([[0.0, 0.0], [1.0, 0.0]])
    d2 = KMeansScratch._pairwise_distances(X, C)
    assert d2.shape == (2, 2)
    # exact squared distances
    assert np.isclose(d2[0, 0], 0.0)
    assert np.isclose(d2[0, 1], 1.0)
    assert np.isclose(d2[1, 0], 1.0)
    assert np.isclose(d2[1, 1], 0.0)


def test_fit_converges_and_predict():
    X, _ = make_simple_blobs()
    km = KMeansScratch(n_clusters=3, random_state=0, max_iters=200, tol=1e-4)
    km.fit(X)
    assert km.centroids_ is not None
    assert km.labels_ is not None
    assert km.n_iter_ is not None
    preds = km.predict(X[:5])
    assert preds.shape == (5,)


def test_inertia_reasonable():
    X, _ = make_simple_blobs()
    km = KMeansScratch(n_clusters=3, random_state=0)
    km.fit(X)
    # inertia should be positive and not absurdly large
    assert km.inertia_ > 0
    assert km.inertia_ < 1e6


def test_close_to_true_centers():
    X, true_C = make_simple_blobs(n=600, centers=3, seed=7)
    km = KMeansScratch(n_clusters=3, random_state=7)
    km.fit(X)
    # match learned centroids to true centers (greedy by distance)
    learned = km.centroids_
    # For each learned centroid, find nearest true center and compute mean distance
    dists = []
    for j in range(3):
        d = np.linalg.norm(true_C - learned[j], axis=1).min()
        dists.append(d)
    assert np.mean(dists) < 0.6  # loose bound due to randomness
