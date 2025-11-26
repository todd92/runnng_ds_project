from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import sys
from typing import Dict, Any

# ==========================================
# 1. PASTE YOUR KMEANS CLASS HERE
# ==========================================
# (Copy the entire 'class Kmeans: ...' block from your other script and paste it here)
# If you don't do this, the pickle load will fail saying "Can't get attribute 'Kmeans'"

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

import __main__
setattr(__main__, "Kmeans", Kmeans)
# --- Helper Functions (You don't need to change these) ---

# ==========================================
# 2. LOAD THE BRAIN
# ==========================================
try:
    with open("running_model.pkl", "rb") as f:
        package = pickle.load(f)
    
    model = package["model"]
    scaler_mean = package["scaler_mean"]
    scaler_std = package["scaler_std"]
    expected_columns = package["columns"]
    
    print("Model loaded successfully!")
    print(f"Expecting columns: {expected_columns}")

except FileNotFoundError:
    print("Error: running_model.pkl not found.")
    sys.exit(1)
except AttributeError:
    print("Error: Could not load the model. Did you paste the Kmeans class above?")
    sys.exit(1)

app = FastAPI()

# ==========================================
# 3. DEFINE INPUT DATA
# ==========================================
class RunData(BaseModel):
    # We use a flexible dict to accept whatever columns you trained on
    # (distance_miles, duration, hr, etc.)
    data: Dict[str, Any]

@app.get("/")
def home():
    return {"message": "Running Cluster API is Online"}

@app.post("/predict")
def predict_cluster(run: RunData):
    try:
        # 1. Convert incoming JSON to DataFrame
        # We expect the user to send: {"data": {"distance_miles": 5, "maxHR": 150...}}
        df_input = pd.DataFrame([run.data])

        # 2. Align Columns
        # Ensure the input has the EXACT same columns as training.
        # If a column is missing (e.g. user didn't send 'endLatitude'), fill it with 0 (average).
        df_input = df_input.reindex(columns=expected_columns, fill_value=0)

        # 3. Apply the Z-Score Math (Using the saved Mean/Std)
        # This is critical: We must scale using the TRAINING stats, not the input stats.
        X_scaled = (df_input - scaler_mean) / scaler_std
        
        # 4. Predict
        # We convert to numpy because your class expects numpy
        prediction_index = model.predict(X_scaled.values)[0]
        
        # 5. Interpret (Optional: Customize these names based on your k=3 analysis)
        cluster_names = {
            0: "Cluster 0", 
            1: "Cluster 1", 
            2: "Cluster 2"
        }
        
        return {
            "cluster_id": int(prediction_index),
            "cluster_name": cluster_names.get(int(prediction_index), "Unknown"),
            "input_summary": f"{run.data.get('distance_miles', 0)} miles"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))