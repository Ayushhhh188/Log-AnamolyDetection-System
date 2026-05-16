import pandas as pd
import numpy as np
import os
import joblib


# PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "parsed_logs_sample.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("="*60)
print("🔧 FEATURE ENGINEERING")
print("="*60)


# LOAD DATA
print("\n📥 Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")


# FIX TIMESTAMP
print("\nConverting timestamp...")
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df.sort_values(by='timestamp').reset_index(drop=True)
print(f"After cleaning: {len(df)} rows")


# HANDLE NaN VALUES
print("\nHandling NaN values...")
df['component'] = df['component'].fillna('UNKNOWN')
df['log_level'] = df['log_level'].fillna('UNKNOWN')
df['process_id'] = df['process_id'].fillna('UNKNOWN')
df['content'] = df['content'].fillna('EMPTY')
print("NaN values replaced with 'UNKNOWN'")


# CREATE CATEGORY MAPPINGS
print("\n📝 Creating category mappings...")

# Component mapping
component_categories = df['component'].astype('category').cat.categories
component_mapping = {i: cat for i, cat in enumerate(component_categories)}
component_to_code = {cat: i for i, cat in enumerate(component_categories)}
df['component_encoded'] = df['component'].map(component_to_code)

# Log level mapping
log_level_categories = df['log_level'].astype('category').cat.categories
log_level_mapping = {i: cat for i, cat in enumerate(log_level_categories)}
log_level_to_code = {cat: i for i, cat in enumerate(log_level_categories)}
df['log_level_encoded'] = df['log_level'].map(log_level_to_code)

# Process ID mapping
process_categories = df['process_id'].astype('category').cat.categories
process_mapping = {i: cat for i, cat in enumerate(process_categories)}
process_to_code = {cat: i for i, cat in enumerate(process_categories)}
df['process_id_encoded'] = df['process_id'].map(process_to_code)

# Save mappings
joblib.dump(component_mapping, os.path.join(MODEL_DIR, "component_mapping.pkl"))
joblib.dump(log_level_mapping, os.path.join(MODEL_DIR, "log_level_mapping.pkl"))
joblib.dump(process_mapping, os.path.join(MODEL_DIR, "process_mapping.pkl"))

# Save reverse mappings for inference
joblib.dump(component_to_code, os.path.join(MODEL_DIR, "component_to_code.pkl"))
joblib.dump(log_level_to_code, os.path.join(MODEL_DIR, "log_level_to_code.pkl"))
joblib.dump(process_to_code, os.path.join(MODEL_DIR, "process_to_code.pkl"))

print(f"Component mappings: {len(component_mapping)} categories")
print(f"Log level mappings: {len(log_level_mapping)} categories")
print(f"Process ID mappings: {len(process_mapping)} categories")


# CREATE EVENT ID FROM CONTENT
print("\nCreating event_id from content...")
content_categories = df['content'].astype('category').cat.categories
content_to_code = {cat: i for i, cat in enumerate(content_categories)}
df['event_id'] = df['content'].map(content_to_code)

# Save content mapping
joblib.dump(content_to_code, os.path.join(MODEL_DIR, "content_to_code.pkl"))
print(f"Event ID mappings: {len(content_to_code)} unique contents")

# =========================
# TIME FEATURES
# =========================
print("\nExtracting time features...")
df['hour_of_day'] = df['timestamp'].dt.hour
df['time_gap'] = df['timestamp'].diff().dt.total_seconds()
df['time_gap'] = df['time_gap'].fillna(0)
df['time_gap'] = np.clip(df['time_gap'], 0, 60)


# ERROR FLAG
print("\n Creating error flag...")
df['error_flag'] = df['log_level'].apply(lambda x: 0 if x == "INFO" else 1)

# EVENT FREQUENCY
print("\nCalculating event frequency...")
event_counts = df['event_id'].value_counts().to_dict()
df['event_frequency'] = df['event_id'].map(event_counts)

# Rare event flag (bottom 5% frequency)
rare_threshold = np.percentile(df['event_frequency'], 5)
df['rare_event_flag'] = (df['event_frequency'] <= rare_threshold).astype(int)
print(f" Rare event threshold: {rare_threshold:.2f}")


# EVENTS PER MINUTE
print("\nCalculating events per minute...")
df['minute'] = df['timestamp'].dt.floor('min')
events_per_min = df.groupby('minute').size().to_dict()
df['events_per_minute'] = df['minute'].map(events_per_min)


# PROCESS BEHAVIOR
print("\nExtracting process behavior...")
events_per_process = df.groupby('process_id').size().to_dict()
df['events_per_process'] = df['process_id'].map(events_per_process)

unique_events_per_process = df.groupby('process_id')['event_id'].nunique().to_dict()
df['unique_event_types_per_process'] = df['process_id'].map(unique_events_per_process)


# APPLY LOG TRANSFORMATIONS
print("\nApplying log transformations...")
for col in ['event_frequency', 'events_per_process', 'events_per_minute']:
    df[col] = np.log1p(df[col])


# SELECT FINAL FEATURES
print("\nSelecting final features...")
final_features = [
    'component_encoded',
    'error_flag',
    'event_frequency',
    'event_id',
    'events_per_minute',
    'events_per_process',
    'hour_of_day',
    'log_level_encoded',
    'process_id_encoded',
    'rare_event_flag',
    'time_gap',
    'unique_event_types_per_process'
]

# Ensure all features exist
for f in final_features:
    if f not in df.columns:
        print(f"⚠️ Creating missing feature: {f}")
        df[f] = 0

df_features = df[final_features].copy()


# CLEAN AND SAVE
print("\nCleaning data...")
df_features = df_features.replace([np.inf, -np.inf], np.nan)
df_features = df_features.fillna(0)
df_features = df_features.astype(np.float32)

print(f"\nSaving features to {OUTPUT_FILE}...")
df_features.to_csv(OUTPUT_FILE, index=False)

# Save feature columns for inference
joblib.dump(final_features, os.path.join(MODEL_DIR, "feature_columns.pkl"))

# SUMMARY
print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETE!")
print("="*60)
print(f"Features shape: {df_features.shape}")
print(f"Feature columns ({len(final_features)}):")
for i, col in enumerate(final_features):
    print(f"   {i+1}. {col}")
print(f"\nModel directory: {MODEL_DIR}")
print(f" Features saved to: {OUTPUT_FILE}")
print("\nSample values (first row):")
print(df_features.iloc[0].tolist())
print("="*60)