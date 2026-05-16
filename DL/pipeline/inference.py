import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "outputs", "models")


# LOAD MODELS
print("Loading models...")

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoder = tf.keras.models.load_model(os.path.join(MODEL_DIR, "encoder.keras"))
autoencoder = tf.keras.models.load_model(os.path.join(MODEL_DIR, "autoencoder.keras"))
isolation_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
threshold = joblib.load(os.path.join(MODEL_DIR, "if_threshold.pkl"))

print(f"Expected {len(feature_columns)} features")
print(f"Threshold: {threshold:.6f}")

# FEATURE ENGINEERING
def extract_features(log):
    df = pd.DataFrame([log])

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Time features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['time_gap'] = 0

    # Encode categorical (use category codes like training)
    df['component_encoded'] = df['component'].astype('category').cat.codes
    df['log_level_encoded'] = df['log_level'].astype('category').cat.codes
    df['process_id_encoded'] = df['process_id'].astype('category').cat.codes

    # Event ID from content
    df['event_id'] = df['content'].astype('category').cat.codes

    # Default behavioral features
    df['event_frequency'] = 1
    df['rare_event_flag'] = 0
    df['events_per_minute'] = 1
    df['events_per_process'] = 1
    df['unique_event_types_per_process'] = 1

    df['error_flag'] = df['log_level'].apply(lambda x: 0 if x == "INFO" else 1)

    # Apply same transforms as training
    if 'time_gap' in df.columns:
        df['time_gap'] = np.clip(df['time_gap'], 0, 60)

    for col in ['event_frequency', 'events_per_process', 'events_per_minute']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Replace inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Match training features exactly
    df_aligned = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        if col in df.columns:
            df_aligned[col] = df[col].values
        else:
            df_aligned[col] = 0

    return df_aligned


# PREDICTION FUNCTION
def predict_log(log, debug=False, sensitivity='normal'):
    sensitivity_factors = {'low': 1.5, 'normal': 1.0, 'high': 0.7}
    factor = sensitivity_factors.get(sensitivity, 1.0)
    final_threshold = threshold * factor

    # Feature extraction
    features = extract_features(log)
    
    # Scale
    features_scaled = scaler.transform(features)
    features_scaled = np.nan_to_num(features_scaled)

    # latent vector and reconstruction
    latent = encoder.predict(features_scaled, verbose=0)
    reconstructed = autoencoder.predict(features_scaled, verbose=0)
    recon_error = np.mean((features_scaled - reconstructed) ** 2, axis=1, keepdims=True)

    # Combine for Isolation Forest
    final_features = np.hstack([latent, recon_error])

    # Isolation Forest score
    score = isolation_forest.decision_function(final_features)[0]
    label = 1 if score >= final_threshold else -1

    # Severity
    if label == -1:
        if score < final_threshold - 0.05:
            severity = "critical"
        else:
            severity = "suspicious"
    else:
        severity = "normal"

    if debug:
        print(f"\nFeatures scaled shape: {features_scaled.shape}")
        print(f"Latent vector shape: {latent.shape}")
        print(f"Reconstruction error: {recon_error[0][0]:.6f}")
        print(f"Score: {score:.6f}")
        print(f"Threshold: {final_threshold:.6f}")
        print(f"Label: {'ANOMALY' if label == -1 else 'NORMAL'}")

    return {
        "anomaly_score": float(score),
        "label": int(label),
        "severity": severity,
        "threshold": float(final_threshold),
        "reconstruction_error": float(recon_error[0][0])
    }

# TEST RUN
if __name__ == "__main__":
    sample_log = {
        "timestamp": "2015-08-21 11:16:06.804",
        "process_id": "None",
        "log_level": "INFO",
        "component": "org.apache.hadoop.hdfs.server.datanode.DataNode",
        "content": "registered UNIX signal handlers for [TERM, HUP]"
    }

    result = predict_log(sample_log, debug=True, sensitivity='low')
    print("\n🔍 Result:", result)