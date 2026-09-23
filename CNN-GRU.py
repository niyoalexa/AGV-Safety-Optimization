
import numpy as np 
import pandas as pd
import tensorflow as tf                              
from tensorflow.keras import layers, models         
from sklearn.model_selection import train_test_split       
from sklearn.metrics import classification_report 
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns                                
import matplotlib.pyplot as plt   
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Success: GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"GPU GETFORCE RTX READY: {e}")

MAX_RANGE_MM = 3500.0

def load_hybrid_sequences(file_path, window_size=50, step_size=25):
    df = pd.read_csv(file_path)

    dist_cols = [c for c in df.columns if c.startswith('Dist_')]
    all_distances = df[dist_cols].values.astype('float32')

    all_distances = np.where(
        all_distances <= 0,MAX_RANGE_MM,all_distances)
    X_seq = []

    for i in range(0, len(all_distances) - window_size + 1, step_size):
        X_seq.append(all_distances[i:i + window_size])

    X = np.array(X_seq, dtype='float32')
    X = np.clip(X, 0, MAX_RANGE_MM) / MAX_RANGE_MM

    X = X.reshape(X.shape[0],window_size,len(dist_cols),1)

    print(" --- Data Loading Stats ---")
    print(f"Distance Range: 0mm - {MAX_RANGE_MM}mm mapped to [0,1]")
    print(f"Number of Beams detected: {len(dist_cols)}")
    print(f"Total sequences generated: {X.shape[0]}")

    return X

def prepare_motion_target(X_data):
    current_scans = X_data[:, -1, :, 0]
    first_scans   = X_data[:, 0, :, 0]
    deadzone_threshold = 500.0 / MAX_RANGE_MM
    clean_now = np.where(current_scans < deadzone_threshold,1.0,current_scans)
    clean_then = np.where(first_scans < deadzone_threshold,1.0,first_scans)
    current_min = np.min(clean_now, axis=1)
    previous_min = np.min(clean_then, axis=1)
    raw_delta = current_min - previous_min

    motion_target = np.clip((raw_delta * 2.5) + 0.5, 0, 1)

    return motion_target.astype('float32')

X_all = load_hybrid_sequences(
    'Lidar_data_V2.csv',
    window_size=50,
    step_size=22
)

y_motion_all = prepare_motion_target(X_all)

split_idx = int(len(X_all) * 0.8)

X_train = X_all[:split_idx]
X_test  = X_all[split_idx:]

y_train_motion = y_motion_all[:split_idx]
y_test_motion  = y_motion_all[split_idx:]

print("\n--- Final Split Shapes ---")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train_motion: {y_train_motion.shape}")
print(f"y_test_motion : {y_test_motion.shape}")

print("\n--- Motion Target Check ---")
print(f"First training target : {y_train_motion[0]:.6f}")
print(f"Last test target      : {y_test_motion[-1]:.6f}")

print(f"\nMotion target range: "
      f"{y_motion_all.min():.6f} - {y_motion_all.max():.6f}")

def build_fuzzy_extractor(window_size=50, num_angles=1351):
    K.clear_session()
    
    inputs = layers.Input(shape=(window_size, num_angles, 1))
    
    x = layers.GaussianNoise(0.001)(inputs)

    x = layers.Conv2D(64, (3, 4), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 4))(x) 
    x = layers.Conv2D(128, (3, 4), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 4))(x)
    
    current_shape = x.shape
    x = layers.Reshape((window_size, current_shape[2] * current_shape[3]))(x)   

    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x) 
    
    shared = layers.Dense(128, activation='relu')(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.1)(shared)

    motion_out = layers.Dense(1, activation='sigmoid', name='motion')(shared)
    model = models.Model(inputs=inputs, outputs=[motion_out])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004),
        loss='mse',
        metrics={'motion': 'mae'}
    )

    return model

model = build_fuzzy_extractor()
model.summary()

clean_log = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {logs['loss']:.6f} | "
        f"Val Loss: {logs['val_loss']:.6f}"
    )
)

model = build_fuzzy_extractor()

history = model.fit(
    X_train,
    y_train_motion,
    validation_data=(
        X_test,
        y_test_motion
    ),
    epochs=50,
    batch_size=16,
    verbose=1,
    callbacks=[clean_log]
)

model.save("motion_regression_model.keras")
model.save_weights("motion_regression_model.weights.h5"
)

print("\nSuccess!")
print("Model saved as: motion_regression_model.keras")
print("Weights saved as: motion_regression_model.weights.h5")

motion_pred = model.predict(X_test,batch_size=16,verbose=0).flatten()
actual_motion = y_test_motion.flatten()

mse = mean_squared_error(actual_motion, motion_pred)
r2 = r2_score(actual_motion,motion_pred)
corr, _ = pearsonr(actual_motion,motion_pred)


print("\n" + "="*75)
print("MOTION REGRESSION PERFORMANCE")
print("="*75)

print(f"MSE         : {mse:.6f}")
print(f"R²          : {r2*100:.2f}%")
print(f"Correlation : {corr*100:.2f}%")
print(f"Parameters  : {model.count_params():,}")

print("="*75)

print("\n--- Motion Prediction Samples ---")

for i in range(min(10, len(actual_motion))):
    print(f"Sample {i:03d} | "f"Actual: {actual_motion[i]:.6f} | "f"Predicted: {motion_pred[i]:.6f}")
