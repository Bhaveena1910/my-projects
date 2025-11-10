import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import base64 

# --- CONFIGURATION (MUST MATCH TRAINING SCRIPT) ---
N_MELS = 128      
SEGMENT_DURATION = 3.0 
TARGET_SR = 22050 
TARGET_TIME_STEPS = 129 
MODEL_PATH = 'music_director_cnn_classifier1.h5' 

# === CRITICAL: DATASET PATH FOR RECOMMENDATIONS ===
DATASET_ROOT_PATH = r'D:\sem5\ml\mini\WAV' 
# =================================================

# === CRITICAL: HARDCODED DIRECTOR NAMES ===
ACTUAL_DIRECTOR_NAMES = [
    "A R Rahman",       
    "Anirudh Ravichandar",      
    "Ilaiyaraja",       
    # Add all your remaining director names here!
]
# ==========================================

# === NEW: DIRECTOR IMAGES MAPPING ===
DIRECTOR_IMAGES = {
    "A R Rahman": r'C:\Users\Bhave\Downloads\rahman.jpeg', # Placeholder path, replace with actual
    "Anirudh Ravichandar": r'C:\Users\Bhave\Downloads\ani.jpeg',
    "Ilaiyaraja": r'C:\Users\Bhave\Downloads\raja.jpeg',
    # Add all remaining director images here!
}
DEFAULT_DIRECTOR_IMAGE = r'C:\Users\Bhave\Downloads\bg1.jpeg' # Fallback image
# ==========================================

# --- UTILITY CSS & STYLING (Minor adjustment to main-prediction-card) ---

def get_base64_image(image_path):
    """Converts a local image file to a base64 string for CSS embedding."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return None

def inject_custom_css():
    """Injects custom CSS for a more attractive look."""
    
    BACKGROUND_IMAGE_PATH = r'C:\Users\Bhave\Downloads\bg.avif' 
    base64_img = get_base64_image(BACKGROUND_IMAGE_PATH)
    
    bg_style = ""
    if base64_img:
        bg_style = f"""
        background-image: url("data:image/jpg;base64,{base64_img}");
        background-size: cover;
        background-attachment: fixed;
        """
    else:
        bg_style = "background: radial-gradient(circle at top left, #2c3e50 0%, #1a1a1a 100%);"
        st.warning(f"⚠️ Could not load background image from: '{BACKGROUND_IMAGE_PATH}'. Using gradient background.")

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        .stApp {{
            {bg_style}
        }}
        
        html, body, [class*="st-"] {{
            font-family: 'Roboto', sans-serif;
        }}
        .css-1d391kg {{ 
            padding-top: 35px;
            padding-bottom: 35px;
        }}
        .main-prediction-card {{
            background-color: #263340; 
            padding: 15px; /* Slightly reduced padding */
            border-radius: 10px; /* Slightly smaller border radius */
            text-align: center;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3); /* Slightly softer shadow */
            color: white;
            margin-top: 15px; /* Added margin-top to separate from image */
            margin-bottom: 15px; 
            min-height: 100px; /* Reduced min-height */
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .director-name {{
            font-size: 2em; /* Adjusted font size */
            font-weight: 700;
            margin-top: 5px; /* Reduced margin */
            line-height: 1.1;
        }}
        .confidence-score {{
            font-size: 1.2em; /* Adjusted font size */
            font-weight: 400;
            color: #e0e0e0;
        }}
        .stAudio {{
            margin-top: 5px;
            margin-bottom: 10px;
        }}
        
        .catalog-container {{
            overflow-y: auto; 
            max-height: 400px; 
            padding: 10px; 
            border: 1px solid #384555; 
            border-radius: 8px; 
            margin-top: 15px;
            background-color: rgba(38, 51, 64, 0.5);
        }}
        .catalog-container::-webkit-scrollbar {{
            width: 8px; 
        }}
        .catalog-container::-webkit-scrollbar-thumb {{
            background-color: #555; 
            border-radius: 10px;
        }}
        .catalog-container::-webkit-scrollbar-track {{
            background: #333; 
            border-radius: 10px;
        }}
        </style>
        """, unsafe_allow_html=True)

# --- CORE FUNCTIONS (UNCHANGED) ---

@st.cache_resource
def load_model_and_classes():
    """Loads the pre-trained model and checks class consistency."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
    except Exception as e:
        st.error(f"ERROR: Could not load model '{MODEL_PATH}'. Ensure the file is in the same directory.")
        return None, None
    try:
        model_num_classes = model.output_shape[-1]
    except Exception:
        st.error("ERROR: Could not read model's output shape.")
        return None, None
    expected_num_classes = len(ACTUAL_DIRECTOR_NAMES)
    if model_num_classes != expected_num_classes:
        st.error(
            f"CRITICAL ERROR: Class count mismatch! Model output layer expects {model_num_classes} classes, "
            f"but the hardcoded list has {expected_num_classes} names. Please check the `ACTUAL_DIRECTOR_NAMES` list."
        )
        return None, None
    return model, ACTUAL_DIRECTOR_NAMES

def get_available_songs(director_name, dataset_path):
    """
    Fetches available song names and their FULL PATHS for local playback.
    Returns a dictionary: {song_name: full_path}
    """
    FALLBACK_SONGS = {
        f"Classic Hit 1 by {director_name} (Sample)": "placeholder_path", 
        f"Top Track 2 by {director_name} (Sample)": "placeholder_path", 
        f"Fan Favorite 3 by {director_name} (Sample)": "placeholder_path"
    }
    
    if not os.path.isdir(dataset_path):
        return FALLBACK_SONGS

    formatted_names = [
        director_name.lower(),
        director_name.lower().replace(' ', '_'),
        director_name.lower().replace(' ', ''),
    ]

    director_dir = None
    if os.path.isdir(dataset_path):
        for item in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, item)) and item.lower() in formatted_names:
                director_dir = os.path.join(dataset_path, item)
                break
    
    song_catalog = {}
    if director_dir:
        try:
            for filename in os.listdir(director_dir):
                if filename.endswith('.wav'):
                    song_name = os.path.splitext(filename)[0]
                    full_path = os.path.join(director_dir, filename)
                    song_catalog[song_name] = full_path
            
            if song_catalog:
                return song_catalog
        except Exception as e:
             pass 

    return FALLBACK_SONGS

def preprocess_audio_for_prediction(audio_file_path, progress_bar, progress_text):
    """Loads a segment, converts it to Mel-Spectrogram, and normalizes it for the model."""
    try:
        progress_text.text("1/3 Loading Audio...")
        y, sr = librosa.load(
            audio_file_path, 
            sr=TARGET_SR, 
            mono=True, 
            duration=SEGMENT_DURATION
        ) 
        progress_bar.progress(33)
        time.sleep(0.3)

        segment_samples = int(SEGMENT_DURATION * sr)
        if len(y) < segment_samples * 0.95:
            st.warning("Audio file is shorter than 3.0 seconds. Prediction stability may be reduced.")
            
        progress_text.text("2/3 Extracting Mel-Spectrogram Features...")
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mels_db = librosa.power_to_db(mels, ref=np.max)
        
        raw_spectrogram = mels_db.copy() 
        
        if mels_db.shape[1] > TARGET_TIME_STEPS:
            mels_db = mels_db[:, :TARGET_TIME_STEPS] 
        elif mels_db.shape[1] < TARGET_TIME_STEPS:
            pad_width = TARGET_TIME_STEPS - mels_db.shape[1]
            mels_db = np.pad(mels_db, ((0, 0), (0, pad_width)), mode='constant')

        feature = mels_db[..., np.newaxis] 
        mean = np.mean(feature)
        std = np.std(feature)
        normalized_feature = (feature - mean) / (std + 1e-6) 

        progress_bar.progress(66)
        time.sleep(0.3)

        return normalized_feature[np.newaxis, ...], raw_spectrogram
        
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None, None

def plot_mel_spectrogram(spectrogram, sr):
    """Generates a matplotlib plot of the Mel-Spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4)) # Keep matplotlib fig size for internal plot rendering
    img = librosa.display.specshow(
        spectrogram, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel', 
        ax=ax,
        cmap='magma'
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title='Mel-Spectrogram (First 3.0s Segment)')
    # Added height parameter for st.pyplot to better match bar chart height
    st.pyplot(fig, use_container_width=True, clear_figure=True) 
    plt.close(fig) # Important to close figures to prevent memory issues

def get_color_from_confidence(confidence):
    """Returns a hex color based on the confidence percentage."""
    if confidence >= 90:
        return "#34A853"
    elif confidence >= 70:
        return "#FBBC05"
    else:
        return "#EA4335"

# --- MAIN STREAMLIT APPLICATION (FINAL REVISED LAYOUT) ---

def main():
    """Streamlit application main function."""
    
    inject_custom_css()
    
    st.set_page_config(
        page_title="Music Director Classifier",
        page_icon="🎶",
        layout="wide",
        initial_sidebar_state="collapsed" 
    )

    # --- Header and Model Status ---
    st.title("🎧 Music Director Classifier")
    st.markdown("### Upload a `.wav` file to predict the composer's style based on Mel-Spectrogram analysis.")
    
    model, class_names = load_model_and_classes()
    
    if model is None:
        st.stop()
    
    st.info(
        f"""
        **System Ready:** Model initialized with **{len(class_names)}** director names for prediction.
        """
    )
    
    st.divider()

    # --- Session State and Placeholders ---
    if 'playing_path' not in st.session_state:
        st.session_state.playing_path = None
        
    if 'unique_audio_key' not in st.session_state:
        st.session_state.unique_audio_key = None
    
    st.empty() 


    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "**1. Upload a WAV Audio File (Max 10MB recommended)**",
        type=['wav']
    )

    if uploaded_file is not None:
        
        st.markdown("### 2. Audio Input Preview")
        st.audio(uploaded_file, format='audio/wav')

        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        progress_text = progress_container.empty() 

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            
            progress_text.text("3/3 Running CNN Prediction...")
            
            X_predict, raw_spectrogram = preprocess_audio_for_prediction(
                tmp_file_path, 
                progress_bar, 
                progress_text
            )
            progress_bar.progress(85)
            
            if X_predict is not None:
                prediction = model.predict(X_predict, verbose=0)
                
                predicted_index = np.argmax(prediction[0])
                predicted_director = class_names[predicted_index]
                confidence = prediction[0][predicted_index] * 100
                
                progress_bar.progress(100)
                progress_text.success("✅ Prediction Complete! Scroll down for results.")
                time.sleep(0.5)
                
                
                # --- Display Results (REFINED LAYOUT) ---
                st.markdown("## 4. Prediction Result 🎉")
                
                # Balanced columns for better visual distribution
                # Col 1: Predicted Director Image + Card (more compact vertical stack)
                # Col 2: Mel-Spectrogram Plot (generous space as it's a key visual)
                # Col 3: Probability Breakdown (bar chart)
                col_result, col_spec, col_probs = st.columns([1.2, 2.8, 1.0]) # Adjusted ratios

                card_accent_color = get_color_from_confidence(confidence)

                with col_result:
                    st.markdown("### Predicted Composer") 
                    
                    # Image first, filling its container width
                    director_image_path = DIRECTOR_IMAGES.get(predicted_director, DEFAULT_DIRECTOR_IMAGE)
                    try:
                        st.image(
                            director_image_path, 
                            caption=predicted_director, 
                            use_container_width=True 
                        ) 
                    except Exception:
                        st.warning(f"Image not found at: {director_image_path}")
                    
                    # Confidence card directly below the image
                    st.markdown(f"""
                        <div class="main-prediction-card" style="border-left: 5px solid {card_accent_color};">
                            <p class="confidence-score">Confidence Score</p>
                            <p class="director-name" style="color: {card_accent_color};">{confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence > 80:
                        st.balloons()
                        
                with col_spec:
                    st.subheader("Mel-Spectrogram Analysis")
                    # The plot function itself now outputs to st.pyplot with use_container_width
                    plot_mel_spectrogram(raw_spectrogram, TARGET_SR)
                    
                with col_probs:
                    st.subheader("Top Probabilities")
                    
                    top_indices = np.argsort(prediction[0])[::-1]
                    
                    prob_data = []
                    for i in top_indices:
                        prob_data.append({'Director': class_names[i], 'Probability': prediction[0][i]})
                    
                    df_probs = pd.DataFrame(prob_data).set_index('Director').head(5)
                    
                    # Bar chart uses use_container_width
                    st.bar_chart(df_probs, height=350, use_container_width=True) 
                
                st.divider()

                # --- TRACK CATALOG SECTION ---
                st.markdown(f"## 5. Track Catalog: {predicted_director} 🎵")
                
                song_catalog = get_available_songs(predicted_director, DATASET_ROOT_PATH)
                available_songs_names = list(song_catalog.keys())
                
                st.markdown(f"**Showing {len(available_songs_names)} tracks available in the local directory for {predicted_director}:**")
                
                if not available_songs_names:
                    st.warning("No songs found in the specified directory.")
                else:
                    def set_song_to_play(name, path, key):
                        st.session_state.playing_song = name
                        st.session_state.playing_path = path
                        st.session_state.unique_audio_key = key


                    st.markdown('<div class="catalog-container">', unsafe_allow_html=True) 
                    
                    for song_name in available_songs_names:
                        full_path = song_catalog[song_name]
                        
                        current_song_key = f"audio_{song_name.replace(' ', '_')}"

                        col_song, col_button = st.columns([4, 1])
                        
                        is_placeholder = full_path == "placeholder_path"

                        with col_song:
                            style = 'font-weight: bold; color: #4CAF50;' if st.session_state.unique_audio_key == current_song_key else ''
                            st.markdown(f'<p style="margin: 0; padding: 0; {style}">🎵 **{song_name}**</p>', unsafe_allow_html=True)

                        with col_button:
                            button_label = '▶️ Play' if not is_placeholder else '⚠️ Sample'
                            
                            if is_placeholder:
                                st.button(button_label, key=f"play_btn_{song_name}", disabled=True)
                            else:
                                if st.button(button_label, key=f"play_btn_{song_name}", on_click=set_song_to_play, args=(song_name, full_path, current_song_key)):
                                    pass 
                        
                        if st.session_state.unique_audio_key == current_song_key and not is_placeholder:
                            try:
                                with open(full_path, 'rb') as f:
                                    audio_bytes = f.read()
                                
                                st.audio(audio_bytes, format='audio/wav')
                                st.success(f"**Now playing:** {song_name}", icon="🔊")
                                
                            except FileNotFoundError:
                                st.error(f"Error: Could not find audio file at: {full_path}")
                            except Exception as e:
                                st.error(f"An error occurred during playback: {e}")
                                    
                    st.markdown('</div>', unsafe_allow_html=True)

        finally:
            os.unlink(tmp_file_path)
            progress_bar.empty()
            progress_text.empty()
            progress_container.empty()


if __name__ == "__main__":
    main()
