import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Set a random seed for reproducibility
np.random.seed(42)


class KMeans:
    """
    A from-scratch implementation of the K-Means clustering algorithm.
    """

    def __init__(self, k=3, max_iters=100, tol=1e-4, visualize_steps=False):
        """
        Initializes the K-Means algorithm.

        Args:
            k (int): The number of clusters to find.
            max_iters (int): The maximum number of iterations to run.
            tol (float): Tolerance for centroid movement. If centroids move
                         less than this amount, the algorithm has converged.
            visualize_steps (bool): If True, plots each iteration of the algorithm.
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.visualize_steps = visualize_steps

        # 'centroids' will store the final cluster centers
        self.centroids = None
        # 'clusters' will store the cluster assignment for each data point
        self.clusters = None

    def _euclidean_distance(self, p1, p2):
        """Helper function to calculate Euclidean distance."""
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def _initialize_centroids(self, X):
        """
        Initializes centroids by randomly selecting 'k' points from the dataset.

        Args:
            X (np.array): The input data (n_samples, n_features).
        """
        n_samples, _ = X.shape
        # Select k random indices from the data
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        # Set the initial centroids to be these data points
        self.centroids = X[random_indices]

    def _assign_clusters(self, X):
        """
        Assigns each data point in X to the nearest centroid.

        Args:
            X (np.array): The input data (n_samples, n_features).

        Returns:
            np.array: An array of cluster indices (one for each sample).
        """
        n_samples, _ = X.shape
        clusters = np.zeros(n_samples, dtype=int)

        for i, point in enumerate(X):
            # Calculate distance from the point to each centroid
            distances = [
                self._euclidean_distance(point, centroid) for centroid in self.centroids
            ]
            # Assign the point to the cluster with the minimum distance
            clusters[i] = np.argmin(distances)

        return clusters

    def _update_centroids(self, X, clusters):
        """
        Recalculates the centroids as the mean of all points assigned to that cluster.

        Args:
            X (np.array): The input data (n_samples, n_features).
            clusters (np.array): The current cluster assignments for each sample.

        Returns:
            np.array: The new, updated centroids.
        """
        new_centroids = np.zeros((self.k, X.shape[1]))

        for i in range(self.k):
            # Get all points assigned to the current cluster 'i'
            cluster_points = X[clusters == i]

            # If a cluster has points, calculate its new centroid (mean)
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If a cluster is empty, re-initialize its centroid
                # (This is a common way to handle empty clusters)
                random_index = np.random.choice(X.shape[0])
                new_centroids[i] = X[random_index]

        return new_centroids

    def fit(self, X):
        """
        Computes the K-Means clustering.

        Args:
            X (np.array): The input data (n_samples, n_features).
        """
        # 1. Initialize centroids
        self._initialize_centroids(X)

        # --- Visualization Setup ---
        if self.visualize_steps and X.shape[1] == 2:
            plt.figure(figsize=(10, 7))
            plt.ion()  # Turn on interactive mode for plotting
        # -------------------------

        for i in range(self.max_iters):
            # 2. Assign points to clusters
            self.clusters = self._assign_clusters(X)

            # --- Visualize Current Step ---
            if self.visualize_steps and X.shape[1] == 2:
                plt.clf()  # Clear the figure
                self._plot_iteration(X, self.clusters, i + 1)
                plt.pause(0.5)  # Pause for 0.5 seconds
            # ------------------------------

            # 3. Update centroids
            old_centroids = np.copy(self.centroids)
            self.centroids = self._update_centroids(X, self.clusters)

            # 4. Check for convergence
            # Calculate the total movement of all centroids
            centroid_movement = np.sum(
                [
                    self._euclidean_distance(old_centroids[j], self.centroids[j])
                    for j in range(self.k)
                ]
            )

            if centroid_movement <= self.tol:
                print(f"Convergence reached at iteration {i + 1}.")
                break

        # --- Visualization Cleanup ---
        if self.visualize_steps and X.shape[1] == 2:
            plt.ioff()  # Turn off interactive mode

            # Draw the final plot one last time
            plt.clf()
            self._plot_iteration(X, self.clusters, i + 1, is_final=True)
            print("Showing final iteration plot. Close window to continue.")
            plt.show()  # Keep the final plot window open
        # ---------------------------

        print("K-Means fitting completed.")

    def predict(self, X):
        """
        Predicts the closest cluster for new data points.

        Args:
            X (np.array): The new data (n_samples, n_features).

        Returns:
            np.array: The cluster assignments.
        """
        if self.centroids is None:
            raise Exception("Model has not been fitted yet. Call .fit() first.")

        return self._assign_clusters(X)

    def _plot_iteration(self, X, clusters, iteration, is_final=False):
        """
        Helper function to plot a single iteration (assumes plt.clf() was called).
        This does NOT call plt.figure() or plt.show().

        Args:
            X (np.array): The input data (n_samples, 2).
            clusters (np.array): The cluster assignments for this iteration.
            iteration (int): The current iteration number.
            is_final (bool): If True, adjusts title for the final plot.
        """
        if X.shape[1] != 2:
            return  # Only plot 2D data

        cmap = plt.get_cmap("viridis", self.k)

        for i in range(self.k):
            cluster_points = X[clusters == i]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                s=50,
                c=[cmap(i)],
                label=f"Cluster {i + 1}",
            )

        # Plot the centroids that *caused* this assignment
        if self.centroids is not None:
            plt.scatter(
                self.centroids[:, 0],
                self.centroids[:, 1],
                s=200,
                c="red",
                marker="X",
                label="Centroids",
                edgecolors="black",
            )

        title = f"K-Means Clustering - Iteration {iteration}"
        if is_final:
            title = f"K-Means Clustering - Final Result (Iteration {iteration})"

        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        # Handle legend to avoid duplicate entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.grid(True, linestyle="--", alpha=0.6)

    def plot_clusters(self, X, clusters):
        """
        Visualizes the clusters (only works for 2D data).

        Args:
            X (np.array): The input data (n_samples, 2).
            clusters (np.array): The cluster assignments.
        """
        print(X)
        if X.shape[1] != 2:
            print("Warning: Plotting is only supported for 2D data.")
            return

        plt.figure(figsize=(10, 7))
        # Use a colormap for distinct cluster colors
        cmap = plt.get_cmap("viridis", self.k)

        for i in range(self.k):
            # Select data points belonging to cluster 'i'
            cluster_points = X[clusters == i]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                s=50,
                c=[cmap(i)],  # Use cmap to get a color
                label=f"Cluster {i + 1}",
            )

        # Plot the final centroids
        if self.centroids is not None:
            plt.scatter(
                self.centroids[:, 0],
                self.centroids[:, 1],
                s=200,
                c="red",
                marker="X",
                label="Centroids",
                edgecolors="black",
            )

        plt.title("K-Means Clustering Results")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Generate sample data
    # We create 3 distinct blobs of data
    n_samples = 300
    n_features = 2
    centers = 3

    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=1.0,
        random_state=42,
    )

    print(f"Generated data shape: {X.shape}")

    # Plot the raw, unclustered data
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], s=50, c="gray", label="Unclustered Data")
    plt.title("Raw Sample Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # 2. Initialize and fit the K-Means model
    k = 3
    # --- MODIFIED ---
    # Set visualize_steps=True to see the animation
    kmeans = KMeans(k=k, max_iters=100, visualize_steps=True)
    # ----------------
    kmeans.fit(X)

    # 3. Get the cluster assignments
    clusters = kmeans.predict(X)

    # 4. Visualize the results
    # This will now plot a *separate*, static window with the final result,
    # *after* the animation window (if any) is closed.
    print("Plotting final, static cluster data...")
    kmeans.plot_clusters(X, clusters)
