import pickle
import numpy as np
import pandas as pd
import hashlib
import os
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted

def load_scaler(path):
    """Load a pickled scaler."""
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler

def file_info(path):
    """Return file size and hash info."""
    with open(path, "rb") as f:
        data = f.read()
    return {
        "file_path": path,
        "file_size": os.path.getsize(path),
        "md5_hash": hashlib.md5(data).hexdigest()
    }

def get_scaler_attributes(scaler):
    """Extract all relevant scaler attributes."""
    attrs = {}
    for attr in dir(scaler):
        if attr.endswith("_") and not attr.startswith("__"):
            value = getattr(scaler, attr)
            if isinstance(value, np.ndarray):
                attrs[attr] = value.copy()
            else:
                attrs[attr] = value
    return attrs

def compare_attributes(attrs1, attrs2):
    """Compare numeric scaler attributes."""
    all_keys = sorted(set(attrs1.keys()) | set(attrs2.keys()))
    results = []
    for key in all_keys:
        val1, val2 = attrs1.get(key), attrs2.get(key)
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            diff = np.linalg.norm(val1 - val2)
            equal = np.allclose(val1, val2, atol=1e-12)
            results.append((key, equal, diff))
        else:
            equal = val1 == val2
            results.append((key, equal, None))
    df = pd.DataFrame(results, columns=["Attribute", "Equal?", "L2 Difference"])
    return df

def visualize_attribute_differences(attrs1, attrs2):
    """Plot comparison for overlapping numeric attributes."""
    common_keys = [k for k in attrs1.keys() if k in attrs2.keys() and isinstance(attrs1[k], np.ndarray)]
    for key in common_keys:
        plt.figure(figsize=(6,4))
        plt.plot(attrs1[key], label=f"Scaler 1 {key}", marker='o')
        plt.plot(attrs2[key], label=f"Scaler 2 {key}", marker='x')
        plt.title(f"Comparison of '{key}'")
        plt.xlabel("Feature Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

def compare_data_transformations(scaler1, scaler2, n_features=None, n_samples=100):
    """Compare how both scalers transform the same random data."""
    check_is_fitted(scaler1)
    check_is_fitted(scaler2)
    
    # Use the same number of features
    n_features = n_features or len(scaler1.scale_)
    X = np.random.randn(n_samples, n_features)
    
    X1 = scaler1.transform(X)
    X2 = scaler2.transform(X)
    diff = X1 - X2
    mean_abs_diff = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))
    
    print(f"\n🔹 Mean abs difference in transformed data: {mean_abs_diff:.6f}")
    print(f"🔹 Max absolute difference: {max_diff:.6f}")
    
    plt.figure(figsize=(6,4))
    plt.boxplot([X1.flatten(), X2.flatten(), diff.flatten()],
                labels=["Scaler 1 Output", "Scaler 2 Output", "Difference"])
    plt.title("Distribution of Transformed Data")
    plt.ylabel("Value")
    plt.show()

def compare_scalers(pkl1, pkl2):
    """Main comparison pipeline."""
    print("=== Loading Scalers ===")
    scaler1, scaler2 = load_scaler(pkl1), load_scaler(pkl2)
    
    print("\n=== File Info ===")
    info1, info2 = file_info(pkl1), file_info(pkl2)
    df_info = pd.DataFrame([info1, info2], index=["Scaler 1", "Scaler 2"])
    print(df_info, "\n")

    print("=== Comparing Attributes ===")
    attrs1, attrs2 = get_scaler_attributes(scaler1), get_scaler_attributes(scaler2)
    df_attrs = compare_attributes(attrs1, attrs2)
    print(df_attrs, "\n")

    print("=== Visualizing Attribute Differences ===")
    visualize_attribute_differences(attrs1, attrs2)
    
    print("=== Comparing Data Transformations ===")
    compare_data_transformations(scaler1, scaler2)
    
    print("✅ Comparison complete!")

# Example usage:
# compare_scalers("scaler1.pkl", "scaler2.pkl")
