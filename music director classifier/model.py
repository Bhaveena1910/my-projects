import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import librosa 

# --- CONFIGURATION ---
# These parameters define the shape of your extracted audio features (Mel-Spectrograms)
N_MELS = 128    # Height of the Spectrogram (number of Mel bands)
SEGMENT_DURATION = 3.0 # NEW: Duration (in seconds) of each clip used for training
# TIME_STEPS will now be determined by the sample rate and SEGMENT_DURATION
# We will calculate the target TIME_STEPS during feature extraction
TARGET_SR = 22050 # Standard sample rate for consistent feature extraction

INPUT_SHAPE = (N_MELS, 129, 1) # (Height, Width=Time Steps, Channels). 129 is the typical width for 3.0s at 22050Hz.

NUM_CLASSES = 10 
NUM_SAMPLES = 0 
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Training Hyperparameters - ADJUSTED FOR SMALL DATASET
EPOCHS = 100 # Increased epochs for longer training
BATCH_SIZE = 32
LEARNING_RATE = 0.0005 # Reduced learning rate for more stable learning

def load_data_and_extract_features(dataset_path):
    """
    Loads audio data from subfolders, segments each file into 3-second clips,
    extracts Mel-Spectrogram features for each clip, and applies normalization.
    """
    print("--- 1. Data Loading and Feature Extraction (with Segmentation) ---")
    
    X_features = []
    y_raw_labels = []
    class_names = []
    class_to_label = {}
    label_counter = 0

    # Fallback logic remains the same
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        print("Falling back to simulated data. Please check your path!")
        
        NUM_SAMPLES = 1200 
        X = np.random.rand(NUM_SAMPLES, N_MELS, 129, 1).astype(np.float32)
        y_raw = np.random.randint(0, 10, NUM_SAMPLES)
        y_categorical = to_categorical(y_raw, num_classes=10)
        return X, y_categorical, y_raw, [f'Class {i}' for i in range(10)]
    
    # -------------------------------------------------------------------
    # ACTUAL DATA LOADING & FEATURE EXTRACTION (Segmenting entire WAV files)
    # -------------------------------------------------------------------

    # Calculate target number of time steps (129 is correct for 3.0s @ 22050Hz)
    TARGET_TIME_STEPS = int(np.ceil(SEGMENT_DURATION * TARGET_SR / 512)) # hop_length is usually 512
    # Ensure INPUT_SHAPE is updated if TARGET_TIME_STEPS changes
    global INPUT_SHAPE
    INPUT_SHAPE = (N_MELS, TARGET_TIME_STEPS, 1)

    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        
        if os.path.isdir(class_path):
            print(f"Processing class: {class_name} (Label: {label_counter})")
            class_names.append(class_name)
            class_to_label[class_name] = label_counter
            
            for filename in os.listdir(class_path):
                # Only process .wav audio file extension
                if filename.endswith('.wav'): 
                    audio_path = os.path.join(class_path, filename)
                    try:
                        # Load the FULL audio file
                        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True) 
                        
                        # Calculate the length of a segment in samples
                        segment_samples = int(SEGMENT_DURATION * sr)
                        
                        # Iterate through the entire song, creating non-overlapping segments
                        for i in range(0, len(y) - segment_samples + 1, segment_samples):
                            y_segment = y[i:i + segment_samples]
                            
                            # Extract Mel-Spectrogram from the segment
                            mels = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=N_MELS)
                            mels_db = librosa.power_to_db(mels, ref=np.max)
                            
                            # Ensure TIME_STEPS consistency (should match TARGET_TIME_STEPS)
                            if mels_db.shape[1] == TARGET_TIME_STEPS:
                                feature = mels_db[..., np.newaxis] 
                                X_features.append(feature)
                                y_raw_labels.append(label_counter)
                            else:
                                # This handles edge cases where librosa output doesn't perfectly match target
                                # We skip non-standard segments
                                pass
                        
                    except Exception as e:
                        print(f"Skipping {filename} or segment extraction failed. Error: {e}")

            label_counter += 1

    X = np.array(X_features)
    y_raw = np.array(y_raw_labels)
    NUM_CLASSES = len(class_names)
    NUM_SAMPLES = len(X)
    
    # --- FEATURE SCALING (CRITICAL for >80% accuracy) ---
    N, H, W, C = X.shape
    X_flat = X.reshape(N, H * W * C)
    
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    
    X_scaled = X_scaled_flat.reshape(N, H, W, C)
    
    # --- End Scaling ---

    y_categorical = to_categorical(y_raw, num_classes=NUM_CLASSES)

    print(f"\nSuccessfully created {NUM_SAMPLES} segments across {NUM_CLASSES} classes.")
    print(f"Features Shape (X): {X_scaled.shape} (Note: Time Steps is {TARGET_TIME_STEPS})")
    print(f"Classes found: {class_names}")
    
    return X_scaled, y_categorical, y_raw, class_names

def build_cnn_model(input_shape, num_classes, learning_rate):
    """
    Defines a deep 2D Convolutional Neural Network (CNN) for classification.
    """
    print("--- 2. Building 2D CNN Model ---")
    
    model = Sequential([
        # Block 1: Feature Extraction
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Classification Head
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # Output layer
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def plot_training_history(history):
    """
    Generates plots for accuracy and loss over training epochs.
    """
    print("\n--- 5. Visualizing Training History ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#3b82f6')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ef4444')
    ax1.set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot Loss
    ax2.plot(history.history['loss'], label='Train Loss', color='#3b82f6')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#ef4444')
    ax2.set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    # 

def plot_confusion_matrix_result(y_true, y_pred_classes, class_names):
    """
    Generates and plots the confusion matrix for final evaluation.
    """
    print("\n--- 6. Visualizing Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(10, 8)) # Make plot larger for better readability
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar=False, linewidths=.5, linecolor='gray', 
                xticklabels=class_names, yticklabels=class_names) # Use class names
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Music Director', fontsize=12)
    plt.xlabel('Predicted Music Director', fontsize=12)
    plt.show()
    # 

def main():
    """
    Executes the entire machine learning pipeline.
    """
    # -------------------------------------------------------------------
    # !! STEP 1: REPLACE THIS PLACEHOLDER PATH !!
    # This path must lead to the root folder containing subfolders 
    # named after your music directors (e.g., /MyData/DirectorA, /MyData/DirectorB).
    dataset_path = r'D:\sem5\ml\mini\WAV'
    # -------------------------------------------------------------------

    # 1. Data Preparation (now using the real loading logic)
    # The function will now try to load your real songs.
    X, y_categorical, y_raw, class_names = load_data_and_extract_features(dataset_path)
    
    # Check if real data was loaded (i.e., not the fallback)
    # The minimum is now much lower because 50 songs can generate hundreds of segments.
    if X.shape[0] < 50:
        print("\n[CRITICAL WARNING] Data size is too small or still using simulation. Cannot achieve >80% accuracy.")
        if "/path/to/your/" in dataset_path:
             print("Please replace the placeholder dataset_path variable.")
        return

    # Split the data
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y_categorical, y_raw, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    # 2. Model Initialization
    num_classes = len(class_names) # Dynamically set the number of classes
    model = build_cnn_model(INPUT_SHAPE, num_classes, LEARNING_RATE)
    
    # 3. Define Callbacks for Stable Training and High Accuracy
    # Patience remains high to account for the slow, stable learning rate.
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, # Increased patience slightly
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=7, # Increased patience slightly
        min_lr=0.00001,
        verbose=1
    )

    # 4. Training the Model
    print("\n--- 4. Starting Model Training ---")
    print(f"Training on {X_train.shape[0]} segments.")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[early_stopping, reduce_lr],
        verbose=2
    )
    
    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

    # 5. Model Evaluation
    print("\n--- 5. Evaluating Model Performance on Test Set ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[FINAL RESULTS] Test Loss: {loss:.4f} | Test Accuracy: {accuracy * 100:.2f}%")
    
    # Predict and visualize
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # 6. Visualization
    plot_training_history(history)
    plot_confusion_matrix_result(y_raw_test, y_pred_classes, class_names)

    # 7. Save Model
    try:
        model.save('music_director_cnn_classifier1.h5')
        print("Model successfully saved to 'music_director_cnn_classifier1.h5'")
    except Exception as e:
        print(f"Could not save model: {e}")

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    main()
