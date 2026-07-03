

import numpy as np 
import pandas as pd
import tensorflow as tf                              
from tensorflow.keras import layers, models         
from sklearn.model_selection import train_test_split 
#from sklearn.metrics import confusion_matrix         
from sklearn.metrics import classification_report    
import seaborn as sns                                
import matplotlib.pyplot as plt   
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
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
    
    # 1. Identify Distance columns (Dist_0 to Dist_1350)
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

    # Reshape for CNN-GRU: (Samples, TimeSteps, Features, Channels)
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
    current_scans = X_data[:, -1, :, 0]  # Shape: (Samples, 1351)
    first_scans = X_data[:, 0, :, 0]     # Shape: (Samples, 1351)
    num_beams = current_scans.shape[1] 

    deadzone_threshold = 500.0 / 3500.0

    clean_now = np.where(current_scans < deadzone_threshold, 1.0, current_scans)
    clean_then = np.where(first_scans < deadzone_threshold, 1.0, first_scans)
    # -----------------------------------------

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

def build_fuzzy_extractor(window_size=50, num_angles=1351):
    K.clear_session()
    
    inputs = layers.Input(shape=(window_size, num_angles, 1))
    
    # 1. Input Noise
    x = layers.GaussianNoise(0.001)(inputs)

    # 2. 2D CNN Backbone
    x = layers.Conv2D(64, (3, 4), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 4))(x) # Keep time resolution, reduce angle dimension
    
    x = layers.Conv2D(128, (3, 4), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 4))(x)
    
    # 3. Transition to RNN
    current_shape = x.shape
    x = layers.Reshape((window_size, current_shape[2] * current_shape[3]))(x)   #flatten
    
    # 4. Temporal Logic (GRU)
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x) 
    
    shared = layers.Dense(128, activation='relu')(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.1)(shared)

    # 6. Multi-Output Heads
    dist_out = layers.Dense(1, activation='sigmoid', name='distance')(shared)
    index_out = layers.Dense(1, activation='sigmoid', name='index')(shared)
    motion_out = layers.Dense(1, activation='sigmoid', name='motion')(shared)

    model = models.Model(inputs=inputs, outputs=[dist_out, index_out, motion_out])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics={'distance': 'mae', 'index': 'mae', 'motion': 'mae'}
    )

    return model

model = build_fuzzy_extractor()
model.summary()


clean_log = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(
        f" Epoch {epoch+1:02d} | Train Loss: {logs['loss']:.6f} | Val Loss: {logs['val_loss']:.6f}"
    )
)

model = build_fuzzy_extractor()
#model.optimizer.learning rate.assign(0.0005)

print("\n--- Starting Stabilized Training Protocol ---")
history = model.fit(
    X_train,
    {
        'distance': y_train_phys[:, 0],
        'index': y_train_phys[:, 1],
        'motion': y_train_phys[:, 2]
    },
    validation_data=(X_test, {
        'distance': y_test_phys[:, 0],
        'index': y_test_phys[:, 1],
        'motion': y_test_phys[:, 2]
    }),
    epochs=50,
    batch_size=16,
    verbose=1,
    callbacks=[clean_log] 
)


#######




def plot_model_performance(history, y_true, y_pred_list):
    # 1. Setup the data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    y_pred_index = y_pred_list[1].flatten() 
    y_true_index = y_true[:, 1]
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r--', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'b-', label='Validation Loss', linewidth=2)
    plt.title('Global Model Convergence (Loss)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
