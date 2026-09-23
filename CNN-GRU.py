
import numpy as np 
import pandas as pd
import tensorflow as tf                              
from tensorflow.keras import layers, models         
from sklearn.model_selection import train_test_split      
from sklearn.metrics import classification_report 
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import layers, models, backend as K
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
    total_rows = len(df)
    
    dist_cols = [c for c in df.columns if c.startswith('Dist_')]
    all_distances = df[dist_cols].values.astype('float32')

    all_distances = np.where(all_distances <= 0, MAX_RANGE_MM, all_distances)
    
    all_labels = df.iloc[:, 6].values
    
    X_seq = []
    y_warning_seq = []
    
    for i in range(0, len(all_distances) - window_size + 1, step_size):
        window = all_distances[i : i + window_size]
        X_seq.append(window)
        y_warning_seq.append(all_labels[i + window_size - 1])
    
    X = np.array(X_seq)
    X = np.clip(X, 0, MAX_RANGE_MM) / MAX_RANGE_MM

    X = X.reshape(X.shape[0], window_size, len(dist_cols), 1)

    print(f"--- Data Loading Stats ---")
    print(f"Distance Range: 0mm - {MAX_RANGE_MM}mm mapped to [0, 1]")
    print(f"Number of Beams detected: {len(dist_cols)}")
    print(f"Total sequences generated: {X.shape[0]}")
    
    return X, np.array(y_warning_seq).astype('float32')

def prepare_physical_targets(X_data):
    """
    Derives normalized targets [0, 1] from the sequence data.
    """
    current_scans = X_data[:, -1, :, 0]  
    first_scans = X_data[:, 0, :, 0]     
    num_beams = current_scans.shape[1] 

    deadzone_threshold = 500.0 / 3500.0

    clean_now = np.where(current_scans < deadzone_threshold, 1.0, current_scans)
    clean_then = np.where(first_scans < deadzone_threshold, 1.0, first_scans)
   

    min_dist = np.min(clean_now, axis=1)

    min_idx = np.argmin(clean_now, axis=1) / (num_beams - 1)

    prev_min = np.min(clean_then, axis=1)
    raw_delta = min_dist - prev_min

    motion_delta = np.clip((raw_delta * 2.5) + 0.5, 0, 1)

    return np.column_stack([min_dist, min_idx, motion_delta]).astype('float32')

X_all, y_warning_all = load_hybrid_sequences('Lidar_data_V2.csv', window_size=50, step_size=22)
y_phys_all = prepare_physical_targets(X_all) # Assumed shape: [Samples, 3]

split_idx = int(len(X_all) * 0.8)

X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train_phys, y_test_phys = y_phys_all[:split_idx], y_phys_all[split_idx:]

def build_model_extractor(window_size=50, num_angles=1351):
    K.clear_session()
    
    inputs = layers.Input(shape=(window_size, num_angles, 1))
    
    x = layers.GaussianNoise(0.001)(inputs)

    x = layers.Conv2D(64, (3, 4), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 4))(x) # Keep time resolution, reduce angle dimension
    
    x = layers.Conv2D(128, (3, 4), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 4))(x)
    
    current_shape = x.shape
    x = layers.Reshape((window_size, current_shape[2] * current_shape[3]))(x)   #flatten
    
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x) 
    
    shared = layers.Dense(128, activation='relu')(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.1)(shared)

    dist_out = layers.Dense(1, activation='sigmoid', name='distance')(shared)
    index_out = layers.Dense(1, activation='sigmoid', name='index')(shared)
    motion_out = layers.Dense(1, activation='sigmoid', name='motion')(shared)

    model = models.Model(inputs=inputs, outputs=[dist_out, index_out, motion_out])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004),
        loss='mse',
        metrics={'distance': 'mae', 'index': 'mae', 'motion': 'mae'}
    )

    return model

model = build_model_extractor()
model.summary()

clean_log = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(
        f" Epoch {epoch+1:02d} | "
        f"Train Loss: {logs['loss']:.6f} | "
        f"Val Loss: {logs['val_loss']:.6f}"
    )
)

history = model.fit(
    X_train,
    {
        'distance': y_train_phys[:, 0],
        'index': y_train_phys[:, 1],
        'motion': y_train_phys[:, 2]
    },
    validation_data=(
        X_test,
        {
            'distance': y_test_phys[:, 0],
            'index': y_test_phys[:, 1],
            'motion': y_test_phys[:, 2]
        }
    ),
    epochs=50,
    batch_size=8,
    verbose=1,
    callbacks=[clean_log]
)

model.save('final_production_model.keras')
print("\nSuccess! Model saved as 'final_production_model.keras'")

raw_preds = model.predict(X_test, batch_size=16, verbose=0)

predictions = np.column_stack([
    raw_preds[0].flatten(),   
    raw_preds[1].flatten(),   
    raw_preds[2].flatten()    
])

actuals = y_test_phys

names = ["Distance", "Index", "Motion"]

print(f"\nFinal Check - Sample 1 Target (Normalized):")
print(f"Distance: {y_test_phys[-1][0]:.6f}")  
print(f"Location: {y_test_phys[-1][1]:.6f}")
print(f"Motion:   {y_test_phys[-1][2]:.6f}")

print("\n--- Final Split Shapes ---")
print(f"X_train shape: {X_train.shape}  -> (Samples, Time, Beams, Channels)")
print(f"y_train_phys:  {y_train_phys.shape}    -> (Samples, [Min_Dist, Min_Idx, Delta])")

print("\n--- Value Check (Sample 1) ---")
print(f"Targets: {y_train_phys[0]}")

print("\n" + "="*75)
print("PERFORMANCE EVALUATION")
print("="*75)

for i, name in enumerate(names):
    mse = mean_squared_error(actuals[:, i],predictions[:, i])
    r2 = r2_score(actuals[:, i],predictions[:, i])
    corr, _ = pearsonr(actuals[:, i],predictions[:, i])

    print(f"{name:<10} | "f"MSE: {mse:.6f} | "f"R²: {r2*100:.2f}% | "f"Correlation: {corr*100:.2f}%")

overall_mse = mean_squared_error(actuals,predictions)
overall_r2 = r2_score(actuals,predictions,multioutput="uniform_average")

print("-"*75)
print(f"Overall MSE : {overall_mse:.6f}")
print(f"Overall R²  : {overall_r2*100:.2f}%")
print(f"Parameters  : {model.count_params():,}")
print("="*75)
