import numpy as np
import queue
import threading
import tensorflow_hub as hub
import urllib.request
import os
import ffmpeg
import wave
from dotenv import load_dotenv
import datetime
from pathlib import Path

# --- LOAD USER CONFIGURATION FROM .env ---
# Always load .env from the directory where main.py is located
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

RTSP_URL = os.getenv("RTSP_URL")
if not RTSP_URL:
    raise ValueError("RTSP_URL is not set. Please check your .env file.")
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

# Find all class indices for dog barks and human speech
DOG_BARK_LABELS = [i for i, name in enumerate(class_names) if 'dog' in name.lower() or 'bark' in name.lower()]
HUMAN_SPEECH_LABELS = [i for i, name in enumerate(class_names) if 'speech' in name.lower() or 'talking' in name.lower() or 'conversation' in name.lower() or 'human voice' in name.lower()]


def detect_audio_events(audio_chunk, sr):
    """
    Run the YAMNet model on an audio chunk and print detected events (dog bark, human speech).
    Args:
        audio_chunk (np.ndarray): The audio data as a 1D numpy array.
        sr (int): The sample rate of the audio data.
    Returns:
        str: Detected event type ('dog_bark', 'human_speech', or None)
    """
    # YAMNet expects mono, float32, 16kHz audio. Resample if needed.
    if sr != 16000:
        import librosa
        audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=16000)
    audio_chunk = audio_chunk.astype('float32')
    scores, embeddings, spectrogram = yamnet_model(audio_chunk)
    scores_np = scores.numpy()
    mean_scores = scores_np.mean(axis=0)
    # Check for dog bark
    for idx in DOG_BARK_LABELS:
        if mean_scores[idx] > 0.1:
            print(f"Detected: {class_names[idx]} (score: {mean_scores[idx]:.2f}) [DOG BARK]")
            return 'dog_bark'
    # Check for human speech
    # for idx in HUMAN_SPEECH_LABELS:
    #     if mean_scores[idx] > 0.1:
    #         print(f"Detected: {class_names[idx]} (score: {mean_scores[idx]:.2f}) [HUMAN SPEECH]")
    #         return 'human_speech'
    return None

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

# Directory to save dog bark audio clips
RECORDINGS_DIR = "dog_bark_recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

def save_audio_clip(audio_chunk, sample_rate):
    """
    Save a numpy float32 audio chunk as a WAV file in the recordings directory.
    Args:
        audio_chunk (np.ndarray): The audio data as a 1D numpy array.
        sample_rate (int): The sample rate of the audio data.
    """
    # Use current date and time for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(RECORDINGS_DIR, f"dog_bark_{timestamp}.wav")
    # Convert float32 [-1, 1] to int16 for WAV
    audio_int16 = np.int16(np.clip(audio_chunk, -1, 1) * 32767)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    print(f"Saved dog bark audio clip: {file_path}")

# --- MAIN DETECTION LOOP ---
def main():
    """
    Main loop: starts the audio extraction thread and runs audio event detection on each audio chunk.
    """
    audio_queue = queue.Queue()  # Thread-safe queue for audio data
    # Start the audio extraction worker in a background thread
    threading.Thread(target=rtsp_audio_stream_worker, args=(audio_queue,), daemon=True).start()
    print("Dog bark and human speech detection started.")
    while True:
        # Wait for the next audio chunk from the queue
        audio_chunk = audio_queue.get()
        # Run the detection function
        event = detect_audio_events(audio_chunk, SAMPLE_RATE)
        if event == 'dog_bark':
            save_audio_clip(audio_chunk, SAMPLE_RATE)
        # You can add more actions for 'human_speech' if desired

if __name__ == "__main__":
    main()