import numpy as np
import pandas as pd
import random

# These are just here from your original code, they aren't used by the class
first_clustering = np.array([[1, 3], [1, 4], [1, 0], [10, 3], [10, 4], [10, 0]])
second_clustering = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
np.sqrt(np.sum((first_clustering[0] - first_clustering[1]) ** 2))


class Kmeans:
    """
    A from-scratch implementation of the K-Means clustering algorithm.
    """

    def __init__(self, k=3, max_iters=100, random_state=42):
        """
        Initializes the K-Means classifier.

        Args:
            k (int): The number of clusters to form.
            max_iters (int): The maximum number of iterations to run.
            random_state (int): Seed for reproducibility.
        """
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state

        # Set random seed for reproducibility
        if self.random_state:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        # List of cluster centers (centroids)
        self.centroids = []

        # A list of lists, where each inner list contains the indices
        # of the data points belonging to that cluster.
        self.clusters = [[] for _ in range(self.k)]

    def _euclidean_distance(self, p1, p2):
        """
        Calculates the Euclidean distance between two points (p1 and p2).
        """
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def _initialize_centroids(self, X):
        """
        Initializes and returns the first set of k centroids.
        """
        n_samples = len(X)
        # Get k random, unique indices
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        # Select the points corresponding to those indices
        centroids = X[random_indices]
        print("Initializing centroids...")
        return centroids

    def _assign_clusters(self, X):
        """
        Assigns each data point in X to the nearest centroid.
        """
        print("Assigning points to clusters...")
        clusters = [[] for _ in range(self.k)]

        for idx, point in enumerate(X):
            # Calculate distance from this point to each centroid
            distances = [self._euclidean_distance(point, c) for c in self.centroids]
            # Find the index of the closest centroid
            closest_centroid_idx = np.argmin(distances)
            # Assign the point's index to that cluster
            clusters[closest_centroid_idx].append(idx)

        return clusters

    def _update_centroids(self, X):
        """
        Calculates and returns the new centroids as the mean
        of all points in each cluster.
        """
        new_centroids = np.zeros_like(self.centroids)

        for cluster_idx, cluster in enumerate(self.clusters):
            # Check if the cluster is not empty
            if cluster:
                # Get all the actual data points for this cluster
                cluster_points = X[cluster]
                # Calculate the mean of those points (the new centroid)
                new_mean = np.mean(cluster_points, axis=0)
                new_centroids[cluster_idx] = new_mean
            else:
                # Handle empty cluster: re-initialize it to a random point
                # This prevents the cluster from "dying"
                print(
                    f"Warning: Cluster {cluster_idx} is empty. Re-initializing centroid."
                )
                new_centroids[cluster_idx] = X[np.random.randint(len(X))]

        print("Updating centroids...")
        return new_centroids

    def fit(self, X):
        """
        Computes the K-Means clustering.
        """
        print("Starting K-Means clustering...")

        # 1. Initialize centroids
        self.centroids = self._initialize_centroids(X)

        # 2. Start the main loop
        for i in range(self.max_iters):
            # 3. Assign points to clusters
            self.clusters = self._assign_clusters(X)

            # 4. Store the old centroids to check for convergence
            old_centroids = self.centroids.copy()

            # 5. Calculate new centroids
            self.centroids = self._update_centroids(X)

            # 6. Check for convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged after {i + 1} iterations.")
                break
        else:
            # This 'else' block runs if the 'for' loop completes without 'break'
            print(f"Did not converge after {self.max_iters} iterations.")

        print("Clustering complete.")

    def predict(self, X):
        """
        Predicts the cluster for each data point in X.
        """
        labels = np.zeros(len(X))

        for idx, point in enumerate(X):
            # Calculate distances from the point to all *final* centroids
            distances = [self._euclidean_distance(point, c) for c in self.centroids]
            # Find the index of the closest centroid
            closest_idx = np.argmin(distances)
            # Assign that index to the labels array
            labels[idx] = closest_idx

        print("Predicting labels...")
        return labels.astype(int)


# --- Helper Functions (You don't need to change these) ---


def load_data_from_csv(filepath):
    """
    Helper function to load data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            print(f"Error: No numeric columns found in {filepath}")
            return None
        df_filled = df_numeric.fillna(df_numeric.mean())
        print(f"Successfully loaded and preprocessed data from {filepath}.")
        return df_filled.values
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None


def create_sample_data():
    """Helper function to create some sample 2D data for testing."""
    print("Generating sample data...")
    cluster1 = np.random.randn(50, 2) + np.array([0, -5])
    cluster2 = np.random.randn(50, 2) + np.array([5, 5])
    cluster3 = np.random.randn(50, 2) + np.array([-5, 5])
    X = np.vstack([cluster1, cluster2, cluster3])
    np.random.shuffle(X)
    return X


# --- Main execution block (Your testing ground) ---
if __name__ == "__main__":
    # --- THIS IS THE FIX ---
    # Initialize X to None so the check below works
    X = None
    # ----------------------

    # --- Option 1: Load your custom data ---
    # Uncomment these lines to load from a CSV
    # FILE_PATH = 'your_file.csv'
    # X = load_data_from_csv(FILE_PATH)

    # --- Option 2: Use sample data ---
    # If X is still None (either because Option 1 was skipped
    # or because loading failed), create sample data.
    if X is None:
        X = create_sample_data()

    if X is not None:
        # --- Run the K-Means Algorithm ---
        K_VALUE = 3

        kmeans = Kmeans(k=K_VALUE, max_iters=100, random_state=42)

        # This is where your `fit` method will be called
        kmeans.fit(X)

        # This is where your `predict` method will be called
        labels = kmeans.predict(X)

        # --- Print Results ---
        if (
            labels is not None
            and kmeans.centroids is not None
            and len(kmeans.centroids) > 0
        ):
            print("\n--- Clustering Results ---")
            print(f"Total data points: {len(X)}")
            print(f"Number of clusters (k): {K_VALUE}")

            print("\nFinal Centroids:")
            for i, centroid in enumerate(kmeans.centroids):
                print(f"   Cluster {i}: {np.round(centroid, 2)}")

            print("\nCluster Assignments (first 10 points):")
            # Ensure we don't go out of bounds if X has fewer than 10 points
            for i in range(min(10, len(X))):
                print(f"   Point {np.round(X[i], 2)} -> Cluster {labels[i]}")
        else:
            print("\nFit/Predict methods not yet implemented. Run complete.")

    else:
        print("Could not run clustering as no data was loaded.")
