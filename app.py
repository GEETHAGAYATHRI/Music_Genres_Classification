#Streamlit Code:
import streamlit as st 
import librosa 
import numpy as np 
from tensorflow.keras.models import load_model 
 
# Function to extract features from audio file 
def extract_features(audio_file): 
    # Load audio file 
    y, sr = librosa.load(audio_file, sr=None) 
    # Extract features (e.g., MFCCs) 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) 
    features = np.mean(mfccs.T, axis=0) 
    return features 
 
# Function to predict genre 
def predict_genre(audio_file): 
    # Load deep learning model 
    model = load_model('music_genre_model.h5') 
    # Extract features from audio file 
    features = extract_features(audio_file) 
    # Reshape features for prediction 
    features = np.expand_dims(features, axis=0) 
    # Predict genre 
    prediction = model.predict(features) 
    # Get predicted genre label 
    genre_label = ['blues', 'country', 'disco','reggae', 'Classical', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 
'Rock'][np.argmax(prediction)] 
    return genre_label 
 
# Streamlit app 
17 
 
def main(): 
    st.title('Music Genre Classification') 
    st.markdown('Upload an audio file and click **Predict** to classify its genre.') 
 
    # File uploader for uploading audio file 
    audio_file = st.file_uploader('Upload Audio File', type=['mp3', 'wav']) 
 
    if audio_file is not None: 
        st.audio(audio_file, format='audio/wav') 
 
        # Predict button 
        if st.button('Predict'): 
            # Classify genre 
            genre = predict_genre(audio_file) 
            st.success(f'The predicted genre is: {genre}') 
 
if __name__ == '__main__': 
    main() 