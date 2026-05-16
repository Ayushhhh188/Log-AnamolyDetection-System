import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "..", "data", "processed", "features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("AUTOENCODER TRAINING - IMPROVED ARCHITECTURE")
print("="*60)

print("\nLoading features...")
df = pd.read_csv(INPUT_FILE)
print(f"Shape: {df.shape}")

print("\nCleaning data...")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
if constant_cols:
    print(f"Removing constant columns: {constant_cols}")
    df = df.drop(columns=constant_cols)

feature_columns = df.columns.tolist()
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))
print(f"Keeping {len(feature_columns)} features")

print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled = np.nan_to_num(X_scaled)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

X_train, X_val = train_test_split(X_scaled, test_size=0.1, random_state=SEED)

input_dim = X_scaled.shape[1]
latent_dim = max(8, input_dim // 2)
print(f"Input: {input_dim}, Latent: {latent_dim}")

# Build deeper autoencoder
input_layer = layers.Input(shape=(input_dim,))

# Encoder - deeper
x = layers.Dense(256, activation='relu')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)

latent = layers.Dense(latent_dim, activation='linear', name="latent_space")(x)

# Decoder - symmetric
x = layers.Dense(64, activation='relu')(latent)
x = layers.BatchNormalization()(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation='relu')(x)
output = layers.Dense(input_dim, activation='linear')(x)

autoencoder = models.Model(input_layer, output)
encoder = models.Model(input_layer, latent)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae']
)

autoencoder.summary()

# Train with more epochs
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

print("\nTraining autoencoder...")
history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=100,
    batch_size=256,
    shuffle=True,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\nSaving models...")
autoencoder.save(os.path.join(MODEL_DIR, "autoencoder.keras"))
encoder.save(os.path.join(MODEL_DIR, "encoder.keras"))

print("\nGenerating latent features...")
latent_vectors = encoder.predict(X_scaled, batch_size=256, verbose=0)
reconstructions = autoencoder.predict(X_scaled, batch_size=256, verbose=0)
recon_error = np.mean((X_scaled - reconstructions) ** 2, axis=1, keepdims=True)

final_features = np.hstack([latent_vectors, recon_error])
print(f"Final features shape: {final_features.shape}")

encoded_path = os.path.join(BASE_DIR, "..", "..", "data", "processed", "encoded_features.csv")
pd.DataFrame(final_features).to_csv(encoded_path, index=False)

print(f"\nReconstruction Error Stats:")
print(f"   Mean: {recon_error.mean():.6f}")
print(f"   Std: {recon_error.std():.6f}")
print(f"   Min: {recon_error.min():.6f}")
print(f"   Max: {recon_error.max():.6f}")

print("\nAutoencoder training complete!")