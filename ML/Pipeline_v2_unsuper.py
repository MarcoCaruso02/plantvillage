import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from collections import Counter


# Create a sklearn Pipeline for preprocessing and clustering
def create_pipeline(model_name, model_params, do_scaling, do_pca, n_components):
    steps = []

    # Step 1: Scaling or pass-through (do nothing)
    if do_scaling:
        steps.append(("scaler", StandardScaler()))
    else:
        # FunctionTransformer with identity function just passes data as is
        steps.append(("scaler", FunctionTransformer(lambda x: x)))

    # Step 2: PCA for dimensionality reduction or pass-through
    if do_pca:
        steps.append(("pca", PCA(n_components=n_components)))
    else:
        steps.append(("pca", FunctionTransformer(lambda x: x)))

    # Step 3: Choose clustering model based on model_name and parameters
    if model_name == "KMEANS":
        model = KMeans(**model_params)
    elif model_name == "GMM":
        model = GaussianMixture(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Add clustering step last in the pipeline
    steps.append(("clustering", model))

    # Return the complete pipeline
    return Pipeline(steps)


# Define parameter grid based on dataset and model for hyperparameter tuning
def get_param_grid(model_name, dataset_name):
    # Select PCA components range depending on dataset size/type
    if dataset_name in ["LBP_8_(D0_1).csv", "LBP_8_den(D0_2).csv"]:
        pca_range = [5, 10, 15]
    elif dataset_name in ["LBP_8_12_16(D1_1).csv", "LBP_8_12_16_den(D1_2).csv"]:
        pca_range = [10, 30, 50, 70]
    elif dataset_name in ["LBP_max_GLCM(D2_1).csv", "LBP_max_GLCM(D2_2).csv"]:
        pca_range = [30, 50, 70, 90]
    else:
        pca_range = [80, 100, 120, 140, 160, 180]

    # Set model-specific hyperparameters to try
    if model_name == "KMEANS":
        model_params = {
            "n_clusters": [8],
            "n_init": [10, 20],
            "random_state": [42]
        }
    elif model_name == "GMM":
        model_params = {
            "n_components": [8],
            "covariance_type": ["full", "tied", "diag", "spherical"],
            "random_state": [42]
        }
    else:
        raise ValueError("Unsupported model")

    # Build the full parameter grid including scaling and PCA toggles
    param_grid = []
    for mp in ParameterGrid(model_params):
        for scale in [True, False]:
            for use_pca in [True, False]:
                if use_pca:
                    # If PCA is enabled, test all values in pca_range for n_components
                    for comp in pca_range:
                        param_grid.append({
                            "model_params": mp,
                            "do_scaling": scale,
                            "do_pca": True,
                            "n_components": comp
                        })
                else:
                    # If PCA disabled, n_components is None
                    param_grid.append({
                        "model_params": mp,
                        "do_scaling": scale,
                        "do_pca": False,
                        "n_components": None
                    })
    return param_grid


# Helper function to convert nested dicts/lists to immutable tuples to allow hashing in Counter
def dict_to_tuple(d):
    if isinstance(d, dict):
        # Sort keys to get consistent ordering, then convert each item recursively
        return tuple(sorted((k, dict_to_tuple(v)) for k, v in d.items()))
    elif isinstance(d, list):
        return tuple(dict_to_tuple(i) for i in d)
    else:
        # Base case: return item as is (assumed hashable)
        return d


# === MAIN SCRIPT START ===

print("\nDataset Loading...\n")

dataset_name = "LBP_max_GLCM_Gabor_den(D4_2).csv"
model_name = "GMM"
#KMEANS, GMM
report_path = "D4_2-GMM.txt"

# Load dataset into pandas DataFrame
df_all = pd.read_csv(dataset_name)

# Specify the label column (target) and separate features
target_column = "label"
data = df_all.drop(columns=[target_column])

# Outer 5-fold cross-validation setup with fixed random state for reproducibility
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store silhouette scores for each test fold
scores = []

# Store the best parameters found in each outer fold
best_param_list = []

# Get parameter grid for hyperparameter search based on model and dataset
param_grid = get_param_grid(model_name, dataset_name)

# Loop over outer folds for train/test split
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
    print(f"\n===== Fold {fold_idx + 1} =====")

    data_train, data_test = data.iloc[train_idx], data.iloc[test_idx]

    best_score = -1  # Initialize best silhouette score for this fold
    best_params = None  # Initialize best hyperparameters for this fold

    # Inner 3-fold CV on training data for hyperparameter tuning
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=fold_idx)

    # Iterate over all hyperparameter combinations in param_grid
    for params in param_grid:
        inner_scores = []  # Keep silhouette scores for inner validation folds

        # Inner fold loop for hyperparameter evaluation
        for inner_train_idx, inner_val_idx in inner_kf.split(data_train):
            X_inner_train = data_train.iloc[inner_train_idx]
            X_inner_val = data_train.iloc[inner_val_idx]

            try:
                # Create pipeline with current hyperparameters
                pipe = create_pipeline(model_name, params["model_params"],
                                       params["do_scaling"],
                                       params["do_pca"],
                                       params["n_components"])

                # Fit model on inner training fold
                pipe.fit(X_inner_train)

                # Predict cluster labels for inner validation fold
                if model_name == "GMM":
                    # For GMM, must transform input with scaler and PCA before predict
                    labels = pipe.named_steps["clustering"].predict(
                        pipe.named_steps["pca"].transform(
                            pipe.named_steps["scaler"].transform(X_inner_val)
                            if params["do_scaling"] else X_inner_val
                        )
                    )
                else:
                    # For other models, pipeline predict is sufficient
                    labels = pipe.predict(X_inner_val)

                # Remove noise label (-1) for DBSCAN if present
                unique_labels = set(labels)
                unique_labels.discard(-1)

                # If too few clusters found, skip this parameter combo
                if len(unique_labels) <= 1:
                    continue

                # Calculate silhouette score on inner validation fold
                score = silhouette_score(X_inner_val, labels)
                inner_scores.append(score)

            except Exception as e:
                # Ignore errors and continue with next inner fold
                continue

        # Calculate average silhouette over inner folds, update best params if improved
        if inner_scores:
            avg_score = np.mean(inner_scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

    # If no valid hyperparameters found, skip fold
    if best_params is None:
        print("No valid configuration found.")
        continue

    best_param_list.append(best_params)
    print(f"Best hyperparameters found: {best_params}")

    # Fit final pipeline on whole training data with best hyperparameters
    best_pipe = create_pipeline(model_name,
                                best_params["model_params"],
                                best_params["do_scaling"],
                                best_params["do_pca"],
                                best_params["n_components"])
    best_pipe.fit(data_train)

    # Predict clusters on the test fold
    if model_name == "GMM":
        labels = best_pipe.named_steps["clustering"].predict(
            best_pipe.named_steps["pca"].transform(
                best_pipe.named_steps["scaler"].transform(data_test)
                if best_params["do_scaling"] else data_test
            )
        )
    else:
        labels = best_pipe.predict(data_test)

    # Remove noise label if any (-1)
    unique_labels = set(labels)
    unique_labels.discard(-1)

    # Compute silhouette score if clustering is valid (more than one cluster)
    if len(unique_labels) > 1:
        final_score = silhouette_score(data_test, labels)
        print(f"[Fold {fold_idx + 1}] Final Silhouette Score: {final_score:.4f}")
        scores.append(final_score)
    else:
        print(f"[Fold {fold_idx + 1}] Invalid clustering (only one cluster)")
        scores.append(-1)  # Mark invalid fold with -1 score

# Calculate average silhouette score across all folds (excluding invalid ones)
print(f"\nAverage Silhouette Score: {np.mean(scores):.4f}")

# Convert dicts in best_param_list to tuples so they can be counted by Counter
configs = [dict_to_tuple(params) for params in best_param_list]

# Find the most frequent configuration across folds
most_common_config_tuple, _ = Counter(configs).most_common(1)[0]

print("Most frequent best configuration:", most_common_config_tuple)

# Write detailed report to text file
with open(report_path, "w") as f:
    f.write("=== MACHINE LEARNING CLASSIFICATION REPORT ===\n\n")
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Model: {model_name}\n\n")
    f.write(f"Best configuration (most frequent): {most_common_config_tuple}\n\n")
    f.write(f"Average Silhouette Score: {np.mean(scores):.4f}\n\n")

    # Write silhouette scores per fold
    for i, score in enumerate(scores, 1):
        f.write(f"Fold {i}: Silhouette Score = {score:.4f}\n")

print(f"Report saved to {report_path}")
