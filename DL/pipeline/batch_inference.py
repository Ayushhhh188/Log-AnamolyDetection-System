import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "outputs", "models")

# LOAD MODELS AND MAPPINGS
print("Loading models...")
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoder = tf.keras.models.load_model(os.path.join(MODEL_DIR, "encoder.keras"))
autoencoder = tf.keras.models.load_model(os.path.join(MODEL_DIR, "autoencoder.keras"))
isolation_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
threshold = joblib.load(os.path.join(MODEL_DIR, "if_threshold.pkl"))

# Load category mappings from training
component_mapping = joblib.load(os.path.join(MODEL_DIR, "component_mapping.pkl"))
log_level_mapping = joblib.load(os.path.join(MODEL_DIR, "log_level_mapping.pkl"))
process_mapping = joblib.load(os.path.join(MODEL_DIR, "process_mapping.pkl"))

# Reverse mappings for encoding
component_to_code = {v: k for k, v in component_mapping.items()}
log_level_to_code = {v: k for k, v in log_level_mapping.items()}
process_to_code = {v: k for k, v in process_mapping.items()}

print(f"Loaded {len(feature_columns)} features")
print(f"Threshold: {threshold:.6f}")


# FEATURE ENGINEERING FOR BATCH
def extract_features_batch(logs_df):
    """
    Extract features for a batch of logs with historical context
    logs_df: DataFrame with columns ['timestamp', 'process_id', 'log_level', 'component', 'content']
    """
    df = logs_df.copy()
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle NaN
    df['component'] = df['component'].fillna('UNKNOWN')
    df['log_level'] = df['log_level'].fillna('UNKNOWN')
    df['process_id'] = df['process_id'].fillna('UNKNOWN')
    
    # Encode using training mappings
    df['component_encoded'] = df['component'].map(component_to_code).fillna(0)
    df['log_level_encoded'] = df['log_level'].map(log_level_to_code).fillna(0)
    df['process_id_encoded'] = df['process_id'].map(process_to_code).fillna(0)
    
    # Event ID
    df['event_id'] = df['content'].apply(lambda x: abs(hash(str(x))) % 100000)
    
    # Time features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['time_gap'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['time_gap'] = np.clip(df['time_gap'], 0, 60)
    
    # Error flag
    df['error_flag'] = df['log_level'].apply(lambda x: 0 if x == "INFO" else 1)
    
    # Calculate historical features from this batch
    if len(df) > 1:
        event_counts = df['event_id'].value_counts().to_dict()
        df['event_frequency'] = df['event_id'].map(event_counts)
        
        rare_threshold = np.percentile(df['event_frequency'], 5)
        df['rare_event_flag'] = (df['event_frequency'] <= rare_threshold).astype(int)
        
        df['minute'] = df['timestamp'].dt.floor('min')
        events_per_min = df.groupby('minute').size().to_dict()
        df['events_per_minute'] = df['minute'].map(events_per_min).fillna(1)
        
        events_per_process = df.groupby('process_id').size().to_dict()
        df['events_per_process'] = df['process_id'].map(events_per_process).fillna(1)
        
        unique_events_per_process = df.groupby('process_id')['event_id'].nunique().to_dict()
        df['unique_event_types_per_process'] = df['process_id'].map(unique_events_per_process).fillna(1)
    else:
        # Single log case - use defaults
        df['event_frequency'] = 1
        df['rare_event_flag'] = 0
        df['events_per_minute'] = 1
        df['events_per_process'] = 1
        df['unique_event_types_per_process'] = 1
    
    # Apply log transforms
    for col in ['event_frequency', 'events_per_process', 'events_per_minute']:
        df[col] = np.log1p(df[col])
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Align with training features
    df_aligned = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        if col in df.columns:
            df_aligned[col] = df[col].values
        else:
            df_aligned[col] = 0
    
    return df_aligned

# BATCH PREDICTION
def predict_batch(logs_df, sensitivity='normal'):
    """
    Predict anomalies for a batch of logs
    logs_df: DataFrame with columns ['timestamp', 'process_id', 'log_level', 'component', 'content']
    """
    sensitivity_factors = {'low': 1.5, 'normal': 1.0, 'high': 0.7}
    factor = sensitivity_factors.get(sensitivity, 1.0)
    final_threshold = threshold * factor
    
    # Extract features
    features = extract_features_batch(logs_df)
    
    # Scale
    features_scaled = scaler.transform(features)
    features_scaled = np.nan_to_num(features_scaled)
    
    # latent vectors and reconstruction errors
    latent = encoder.predict(features_scaled, verbose=0)
    reconstructed = autoencoder.predict(features_scaled, verbose=0)
    recon_error = np.mean((features_scaled - reconstructed) ** 2, axis=1)
    
    # Combine for IF
    final_features = np.hstack([latent, recon_error.reshape(-1, 1)])
    
    # Get scores
    scores = isolation_forest.decision_function(final_features)
    labels = np.where(scores >= final_threshold, 1, -1)
    
    # Severity
    severity = []
    for score, label in zip(scores, labels):
        if label == -1:
            if score < final_threshold - 0.05:
                severity.append("critical")
            else:
                severity.append("suspicious")
        else:
            severity.append("normal")
    
    results = pd.DataFrame({
        'anomaly_score': scores,
        'label': labels,
        'severity': severity,
        'reconstruction_error': recon_error
    })
    
    return results

# TEST
if __name__ == "__main__":
    # Create sample logs with None as process_id
    sample_logs = pd.DataFrame([
        {
            'timestamp': '2015-08-21 11:16:06.804',
            'process_id': None,
            'log_level': 'INFO',
            'component': 'org.apache.hadoop.hdfs.server.datanode.DataNode',
            'content': 'registered UNIX signal handlers for [TERM, HUP]'
        },
        {
            'timestamp': '2015-08-21 11:16:08.169',
            'process_id': None,
            'log_level': 'INFO',
            'component': 'org.apache.hadoop.metrics2.impl.MetricsConfig',
            'content': 'loaded properties from hadoop-metrics.properties'
        },
        {
            'timestamp': '2015-08-21 11:16:08.282',
            'process_id': None,
            'log_level': 'WARN',
            'component': 'org.apache.hadoop.metrics2.impl.MetricsSystemImpl',
            'content': 'Scheduled snapshot period at 10 second(s)'
        },
        {
            'timestamp': '2015-08-21 11:16:08.346',
            'process_id': None,
            'log_level': 'ERROR',
            'component': 'org.apache.hadoop.hdfs.server.datanode.DataNode',
            'content': 'Disk failure detected!'
        }
    ])
    
    print("="*60)
    print("BATCH INFERENCE TEST")
    print("="*60)
    
    results = predict_batch(sample_logs, sensitivity='low')
    
    print("\nResults:")
    print(results)
    
    print(f"\nSummary:")
    print(f"   Normal: {(results['label'] == 1).sum()}")
    print(f"   Anomalies: {(results['label'] == -1).sum()}")
    
    print("\nDetailed Results:")
    for i, row in results.iterrows():
        print(f"   Log {i+1}: Score={row['anomaly_score']:.4f}, Label={row['label']}, Severity={row['severity']}, ReconError={row['reconstruction_error']:.4f}")