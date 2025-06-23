import numpy as np
import queue
import threading
import tensorflow_hub as hub
import urllib.request
import os
import ffmpeg
from dotenv import load_dotenv

# --- LOAD USER CONFIGURATION FROM .env ---
load_dotenv()
RTSP_URL = os.getenv("RTSP_URL")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
CHUNK_DURATION = int(os.getenv("CHUNK_DURATION", 2))
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# --- YAMNet Setup ---
# Load the pre-trained YAMNet model from TensorFlow Hub. YAMNet is an audio event classifier that can detect dog barks and many other sounds.
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(YAMNET_MODEL_HANDLE)

# Download the class labels for YAMNet if not already present. These labels map the model's output indices to human-readable class names.
LABELS_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
LABELS_PATH = 'yamnet_class_map.csv'
if not os.path.exists(LABELS_PATH):
    urllib.request.urlretrieve(LABELS_URL, LABELS_PATH)
# Read the class names from the CSV file (skip the header row)
with open(LABELS_PATH, 'r') as f:
    class_names = [line.strip().split(',')[2] for line in f.readlines()[1:]]

# Find all class indices that mention 'dog' or 'bark' in their label for later detection
DOG_BARK_LABELS = [i for i, name in enumerate(class_names) if 'dog' in name.lower() or 'bark' in name.lower()]


def is_dog_bark(audio_chunk, sr):
    """
    Run the YAMNet model on an audio chunk and check if a dog bark is detected.
    Args:
        audio_chunk (np.ndarray): The audio data as a 1D numpy array.
        sr (int): The sample rate of the audio data.
    Returns:
        bool: True if a dog bark is detected, False otherwise.
    """
    # YAMNet expects mono, float32, 16kHz audio. Resample if needed.
    if sr != 16000:
        import librosa
        audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=16000)
    audio_chunk = audio_chunk.astype('float32')
    # Run the audio through YAMNet. It returns scores for each class for each frame.
    scores, embeddings, spectrogram = yamnet_model(audio_chunk)
    scores_np = scores.numpy()
    mean_scores = scores_np.mean(axis=0)  # Average over time
    # Check if any of the dog bark-related classes have a high enough score
    for idx in DOG_BARK_LABELS:
        if mean_scores[idx] > 0.1:  # Threshold can be tuned for sensitivity
            print(f"Detected: {class_names[idx]} (score: {mean_scores[idx]:.2f})")
            return True
    return False

# --- AUDIO CAPTURE FROM RTSP ---
def rtsp_audio_stream_worker(audio_queue):
    """
    Extract audio from the RTSP stream using ffmpeg and feed it into a queue for processing.
    Args:
        audio_queue (queue.Queue): Thread-safe queue to send audio chunks to the main loop.
    """
    process = (
        ffmpeg
        .input(RTSP_URL, rtsp_transport='tcp', f='rtsp')  # Connect to the RTSP stream using TCP
        .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=SAMPLE_RATE)  # Output raw float32 mono audio at 16kHz
        .run_async(pipe_stdout=True, pipe_stderr=True)  # Run ffmpeg as a subprocess
    )
    print("RTSP audio stream opened. Listening for audio...")
    bytes_per_sample = 4  # float32 = 4 bytes
    chunk_bytes = CHUNK_SIZE * bytes_per_sample  # Number of bytes per audio chunk

    try:
        while True:
            # Read a chunk of audio from ffmpeg's stdout
            in_bytes = process.stdout.read(chunk_bytes)
            if not in_bytes or len(in_bytes) < chunk_bytes:
                print("Stream ended or error.")
                break
            # Convert the raw bytes to a numpy float32 array
            audio_data = np.frombuffer(in_bytes, np.float32)
            # Put the audio chunk into the queue for processing
            audio_queue.put(audio_data)
    finally:
        process.stdout.close()
        process.wait()

# --- MAIN DETECTION LOOP ---
def main():
    """
    Main loop: starts the audio extraction thread and runs dog bark detection on each audio chunk.
    """
    audio_queue = queue.Queue()  # Thread-safe queue for audio data
    # Start the audio extraction worker in a background thread
    threading.Thread(target=rtsp_audio_stream_worker, args=(audio_queue,), daemon=True).start()
    print("Dog bark detection started.")
    while True:
        # Wait for the next audio chunk from the queue
        audio_chunk = audio_queue.get()
        # Run the detection function
        if is_dog_bark(audio_chunk, SAMPLE_RATE):
            print("Dog bark detected!")
        else:
            print("No bark detected in this chunk.")

if __name__ == "__main__":
    main()