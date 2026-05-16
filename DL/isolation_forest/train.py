import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "..", "data", "processed", "encoded_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("ISOLATION FOREST TRAINING")
print("="*60)


# LOAD DATA
print("\nLoading encoded features...")
df = pd.read_csv(INPUT_FILE)
print(f"Shape: {df.shape}")
print(f"First 5 rows (first 3 cols):\n{df.iloc[:5, :3]}")


# CLEAN DATA
print("\nCleaning data...")
X = df.values
X = np.nan_to_num(X)
print(f"Clean shape: {X.shape}")


# TRAIN ISOLATION FOREST
print("\n Training Isolation Forest...")
model = IsolationForest(
    n_estimators=300,
    max_samples='auto',
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

model.fit(X)
print("Model trained")


# SAVE MODEL
print("\nSaving model...")
model_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# =========================
# CALIBRATE THRESHOLD
# =========================
print("\nCalibrating threshold...")
scores = model.decision_function(X)

percentiles_to_try = [1, 2, 5, 10, 15, 20]
print("\nThreshold options:")
for p in percentiles_to_try:
    thr = np.percentile(scores, p)
    anomaly_count = (scores < thr).sum()
    print(f" {p}th percentile: {thr:.6f} -> Would flag {anomaly_count}/{len(scores)} ({anomaly_count/len(scores)*100:.1f}%)")

percentile = 0.25
threshold = np.percentile(scores, percentile)
print(f"\nUsing threshold ({percentile}th percentile): {threshold:.6f}")

joblib.dump(threshold, os.path.join(MODEL_DIR, "if_threshold.pkl"))


# SAVE SCORES
print("\nGenerating anomaly scores...")
labels = model.predict(X)

output_df = pd.DataFrame({
    "anomaly_score": scores,
    "label": labels,
    "is_anomaly": (scores < threshold).astype(int)
})

output_path = os.path.join(BASE_DIR, "..", "..", "outputs", "logs", "anomaly_scores.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)


# SUMMARY
anomaly_count = (scores < threshold).sum()
anomaly_pct = (anomaly_count / len(scores)) * 100

print("\n" + "="*60)
print("ISOLATION FOREST TRAINING COMPLETE!")
print("="*60)
print(f"Total samples: {len(scores)}")
print(f"Anomalies detected: {anomaly_count} ({anomaly_pct:.2f}%)")
print(f"Score range: [{scores.min():.6f}, {scores.max():.6f}]")
print(f"Score mean: {scores.mean():.6f}")
print(f"Threshold: {threshold:.6f}")
print("="*60)