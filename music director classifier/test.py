import numpy as np
import tensorflow as tf
import librosa
import os
import sys

# --- CONFIGURATION (MUST MATCH TRAINING SCRIPT) ---
N_MELS = 128    
SEGMENT_DURATION = 3.0 
TARGET_SR = 22050 
# 129 is the number of time steps (columns) for a 3.0s segment at 22050Hz
TARGET_TIME_STEPS = 129 

# --- MODEL FILE ---
MODEL_PATH = 'music_director_cnn_classifier1.h5'

# === CRITICAL: HARDCODED DIRECTOR NAMES ===
# REPLACE THESE EXAMPLE NAMES WITH YOUR ACTUAL DIRECTOR NAMES.
# They MUST be in the EXACT same order your training folders/classes were processed.
ACTUAL_DIRECTOR_NAMES = [
    "A R Rahman",    # Index 0
    "Anirudh Ravichandar",    # Index 1
    "Ilaiyaraja",    # Index 2
      # Index 5
    # Add all your remaining director names here!
]
# ==========================================


def load_model_and_classes():
    """
    Loads the trained Keras model and uses the hardcoded list of director names.
    It verifies the model's output count matches the length of the hardcoded list.
    """
    try:
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
        print(f"Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Could not load model '{MODEL_PATH}'. Ensure the file is in the correct directory.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- VALIDATION: CHECK CLASS COUNT ---
    try:
        # Get the number of output units from the model
        if len(model.output_shape) == 2:
            model_num_classes = model.output_shape[-1]
        else:
            model_num_classes = model.layers[-1].output_shape[-1]
            
    except Exception:
        print("ERROR: Could not read model's output shape.")
        sys.exit(1)

    expected_num_classes = len(ACTUAL_DIRECTOR_NAMES)
    if model_num_classes != expected_num_classes:
        print(f"\nCRITICAL ERROR: Class count mismatch!")
        print(f"Model output layer expects {model_num_classes} classes, but the 'ACTUAL_DIRECTOR_NAMES' list has {expected_num_classes} names.")
        print("Please ensure your hardcoded list is the exact length of the model's final dense layer.")
        sys.exit(1)
        
    # Return the hardcoded list of real names
    return model, ACTUAL_DIRECTOR_NAMES

def preprocess_audio_for_prediction(audio_path):
    """
    Loads an audio file, extracts the first segment, converts it to a 
    Mel-Spectrogram, and performs normalization.
    """
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at: {audio_path}")
        return None

    try:
        # 1. Load the first 3.0s segment of the audio file
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, duration=SEGMENT_DURATION) 
        
        segment_samples = int(SEGMENT_DURATION * sr)
        if len(y) < segment_samples * 0.95:
            print("ERROR: Audio file is shorter than 3.0 seconds or loading failed.")
            return None
                
        # 2. Extract Mel-Spectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mels_db = librosa.power_to_db(mels, ref=np.max)
        
        # 3. Ensure TIME_STEPS consistency (Should be 129)
        if mels_db.shape[1] > TARGET_TIME_STEPS:
            mels_db = mels_db[:, :TARGET_TIME_STEPS] 
        elif mels_db.shape[1] < TARGET_TIME_STEPS:
            # Pad with zeros if necessary
            pad_width = TARGET_TIME_STEPS - mels_db.shape[1]
            mels_db = np.pad(mels_db, ((0, 0), (0, pad_width)), mode='constant')

        # 4. Reshape and Normalize (Z-Score)
        feature = mels_db[..., np.newaxis] 
        mean = np.mean(feature)
        std = np.std(feature)
        normalized_feature = (feature - mean) / (std + 1e-6) 

        # Add batch dimension: (1, N_MELS, TARGET_TIME_STEPS, 1) for Keras
        return normalized_feature[np.newaxis, ...]
        
    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        return None

def predict_music_director(audio_path, model, class_names):
    """Handles the prediction process for a single audio file."""
    
    print(f"\n--- Starting Prediction for: {os.path.basename(audio_path)} ---")
    
    # 1. Preprocess the audio
    X_predict = preprocess_audio_for_prediction(audio_path)
    if X_predict is None:
        return

    # 2. Make the prediction
    prediction = model.predict(X_predict, verbose=0)
    
    # 3. Interpret the result
    predicted_index = np.argmax(prediction[0])
    predicted_director = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100
    
    # 4. Display the results
    print("----------------------------------------------------------")
    print(f"Predicted Music Director: \033[1m{predicted_director}\033[0m")
    print(f"Confidence: {confidence:.2f}%")
    print("----------------------------------------------------------")
    
    # Optional: print full probabilities
    print("\nFull Probability Breakdown (Top 3):")
    # Sort and display the top 3 predictions
    top_indices = np.argsort(prediction[0])[::-1][:3]
    for i in top_indices:
        name = class_names[i]
        prob = prediction[0][i] * 100
        print(f"- {name}: {prob:.2f}%")


if __name__ == '__main__':
    
    # === SET YOUR INPUT FILE PATH HERE ===
    # IMPORTANT: Change this path to your specific .wav file
    input_audio_path = r'D:\sem5\ml\mini\TEST\thaiya thaiya.wav'
    # ====================================

    # Load model and classes first
    model, class_names = load_model_and_classes() 
    
    # Run prediction
    predict_music_director(input_audio_path, model, class_names)
