import pandas as pd
import numpy as np
import sys
sys.path.append('DL/pipeline')
from batch_inference import predict_batch

# Load some actual logs from your dataset
print("Loading sample logs from training data...")
df = pd.read_csv('data/processed/parsed_logs_sample.csv', nrows=1000)

print(f"Loaded {len(df)} logs")

# Run batch prediction on actual logs
print("\nRunning batch inference on 1000 logs...")
results = predict_batch(df, sensitivity='low')

# Show results
print("\n" + "="*60)
print("RESULTS ON ACTUAL LOGS FROM DATASET")
print("="*60)
print(f"Normal logs: {(results['label'] == 1).sum()}")
print(f"Anomalies: {(results['label'] == -1).sum()}")
print(f"Anomaly percentage: {(results['label'] == -1).sum() / len(results) * 100:.2f}%")

print("\nSeverity breakdown:")
print(results['severity'].value_counts())

# Show first 10 results
print("\nFirst 10 results:")
print(results.head(10))