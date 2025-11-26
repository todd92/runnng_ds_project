import numpy as np
import pandas as pd
import random
import pickle

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

        # 7. Calculate Inertia (Sum of Squared Errors)
        self.inertia_ = 0
        for i, cluster_indices in enumerate(self.clusters):
            if not cluster_indices:
                continue

            # Get all points belonging to this specific cluster
            cluster_points = X[cluster_indices]

            # Get the specific centroid for this cluster
            centroid = self.centroids[i]

            # Calculate squared distance: (Points - Centroid)^2
            # We sum it all up to get the total error for this cluster
            cluster_sum_of_squares = np.sum((cluster_points - centroid) ** 2)

            self.inertia_ += cluster_sum_of_squares

        print(f"Final Inertia: {self.inertia_}")
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
        df_numeric = df_numeric.drop(columns=["activityId"])
        print(df_numeric)
        if df_numeric.empty:
            print(f"Error: No numeric columns found in {filepath}")
            return None
        df_filled = df_numeric.fillna(df_numeric.mean())
        df_scaled = (df_filled - df_filled.mean()) / df_filled.std()
        print(f"Successfully loaded and preprocessed data from {filepath}.")
        return df_scaled.values, df_filled
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

import matplotlib.pyplot as plt

def plot_clusters(X, labels, centroids, k):
    # Assuming X column 0 is Distance and column 3 is MaxHR
    # (Check your load_data function to be sure of column indices!)
    
    # We need to know which index corresponds to which column.
    # Based on your previous printout:
    # Col 0 = distance_miles
    # Col 3 = maxHR (This is a guess, check your specific X structure)
    
    x_col_idx = 6  # Average Cadence
    y_col_idx = 17  # MaxHR
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for the data points
    scatter = plt.scatter(X[:, x_col_idx], X[:, y_col_idx], c=labels, cmap='viridis', alpha=0.6)
    
    # Plot the centroids (Marked with a big Red X)
    plt.scatter(centroids[:, x_col_idx], centroids[:, y_col_idx], 
                s=300, c='red', marker='x', label='Centroids')
    
    plt.title(f'K-Means Clustering (k={k})')
    plt.xlabel('Distance (Scaled)') # Remember these are Z-scores!
    plt.ylabel('Max HR (Scaled)')
    plt.colorbar(scatter, label='Cluster Label')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Main execution block (Your testing ground) ---
if __name__ == "__main__":
    # 1. Load Data
    # Ensure your loader returns BOTH the scaled array (X) and the original df
    FILE_PATH = "/home/toddglad/projects/garmin_repo/kmeans_from_scratch/tests/activity_fact.csv"
    X, df_original = load_data_from_csv(FILE_PATH)

    if X is not None:
        # ==========================================
        # PART 1: The Elbow Method (Find the best k)
        # ==========================================
        print("\n--- Running Elbow Method (k=1 to 10) ---")
        inertias = []
        k_range = range(1, 20)

        for k in k_range:
            # Initialize and fit
            temp_kmeans = Kmeans(k=k, max_iters=100, random_state=42)
            temp_kmeans.fit(X)
            # Store the error score
            inertias.append(temp_kmeans.inertia_)
            print(f"k={k}: Inertia = {temp_kmeans.inertia_:.2f}")

        # (Optional) Copy paste these numbers into Excel or just look for the "Bend"

        # ==========================================
        # PART 2: The Detailed Analysis
        # ==========================================
        print("\n--- Running Detailed Analysis ---")
        # Paste this in your main block to find the index numbers
        # CHANGE THIS NUMBER based on what you see in Part 1
        BEST_K = 3

        print(f"Fitting model with k={BEST_K}...")
        kmeans = Kmeans(k=BEST_K, max_iters=100, random_state=42)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        

        # --- Print Results using the ORIGINAL DataFrame ---
        if labels is not None:
            # Copy the human-readable data
            df_report = df_original.copy()
            df_report["clusters"] = labels

            # Group by cluster to see the averages in REAL units
            cluster_profile = df_report.groupby("clusters").mean()
            print("\nCluster Profiles (Averages):")
            print(cluster_profile)

            # Your specific Deep Dive into Cluster 2
            print("\n--- Cluster 2 Analysis (Long Runs) ---")
            cluster_2 = df_report[df_report["clusters"] == 2]

            # We filter for runs between 6 and 20 miles
            long_runs = cluster_2[
                (cluster_2["distance_miles"] > 6) & (cluster_2["distance_miles"] < 20)
            ]
            print(long_runs)
            plot_clusters(X, labels, kmeans.centroids, BEST_K)
        else:
            print("Prediction failed.")
# ... inside Part 2, after fitting and predicting ...

        print("\n--- Feature Importance Analysis ---")
        
        # 1. Calculate the 'Spread' of the centroids for each column
        #    High Spread = This feature is very different between clusters (IMPORTANT)
        #    Low Spread  = This feature is basically the same in all clusters (USELESS)
        centroid_ranges = np.ptp(kmeans.centroids, axis=0)
        
        # 2. Get the column names so we know what we are looking at
        column_names = df_original.columns
        
        # 3. Zip them together so we can sort them
        features = list(zip(column_names, centroid_ranges))
        
        # 4. Sort by the score (highest first)
        features.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Print the results
        # Paste this in your main block to find the index numbers
        print(f"Index of 'steps': {list(df_original.columns).index('steps')}")
        print(f"Index of 'hrTimeInZone_4': {list(df_original.columns).index('hrTimeInZone_4')}")
        
        # Change these numbers to the two clusters that look close on the plot
        cluster_A = 2 
        cluster_B = 1

# Compare just these two rows from your profile
        comparison = cluster_profile.loc[[cluster_A, cluster_B]]

# Transpose it (swap rows/cols) so it's easier to read
        print(f"\n--- Face-off: Cluster {cluster_A} vs Cluster {cluster_B} ---")
        print(comparison.T)
        for name, score in features:
            print(f"{name}: {score:.4f}")
        
        
        # === PASTE THIS HERE ===
        print("\nSaving model for API...")
        
        # We package everything the API needs into one dictionary
        model_package = {
            "model": kmeans,                    # The trained brain
            "scaler_mean": df_original.mean(),  # The math to scale new data
            "scaler_std": df_original.std(),    # The math to scale new data
            "columns": df_original.columns.tolist() # The order of columns
        }

        with open("running_model.pkl", "wb") as f:
            pickle.dump(model_package, f)

        print("Success! running_model.pkl created.")

    else:
        print("Could not run clustering as no data was loaded.")
