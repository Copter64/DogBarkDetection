import numpy as np
import librosa
from keras.models import load_model
import sys

# Path to your model
MODEL_PATH = 'models/breeds_v1.h5'

# If you have a class list, update this:
breed_names = [
    'not_scotty',
    'scotty',
    # Add more breed names here if your model supports more classes
]

def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.reshape(1, -1)

def print_breed_probs(audio_path):
    model = load_model(MODEL_PATH)
    mfcc_features = extract_mfcc(audio_path)
    pred = model.predict(mfcc_features)[0]
    print("Breed probabilities:")
    for i, prob in enumerate(pred):
        breed = breed_names[i] if i < len(breed_names) else f"class_{i}"
        print(f"  {breed}: {prob:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_breeds.py <audio_file.wav>")
        sys.exit(1)
    audio_path = sys.argv[1]
    print_breed_probs(audio_path)
