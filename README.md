# DogBarkDetection

A Python application that listens to an RTSP audio stream and detects dog barks in real time using a pre-trained YAMNet model from TensorFlow Hub.

---

## Features
- **RTSP Audio Stream**: Connects to an RTSP stream and extracts audio using ffmpeg.
- **Dog Bark Detection**: Uses Google's YAMNet model to classify audio and detect dog barks.
- **Configurable**: User-specific settings (like RTSP URL) are stored in a `.env` file for security and flexibility.

---

## Installation

### 1. Clone the Repository
```powershell
# In your projects directory:
git clone <your-repo-url>  # Or copy the folder if local
cd DogBarkDetection
```

### 2. Set Up Python Virtual Environment (Recommended)
```powershell
# Make sure Python 3.11 is installed
python --version
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Requirements
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install ffmpeg
- Download ffmpeg for Windows: https://ffmpeg.org/download.html
- Extract and add the `bin` folder to your system PATH (so `ffmpeg` is available in the terminal).

---

## Configuration

1. **Create a `.env` file** in the `DogBarkDetection` folder (or copy `.env.sample`):
   ```env
   RTSP_URL=rtsps://YOUR_CAMERA_IP:PORT/STREAM_ID?enableSrtp
   SAMPLE_RATE=16000
   CHUNK_DURATION=2
   ```
   - Replace `RTSP_URL` with your camera's RTSP stream URL.
   - Adjust `SAMPLE_RATE` and `CHUNK_DURATION` if needed.

---

## Usage

1. **Activate your virtual environment (if not already):**
   ```powershell
   .venv\Scripts\activate
   ```
2. **Run the application:**
   ```powershell
   python main.py
   ```
3. **What happens:**
   - The app connects to your RTSP stream and extracts audio using ffmpeg.
   - Audio is processed in real time by the YAMNet model.
   - If a dog bark is detected, a message is printed in the console.

---

## Project Structure
```
DogBarkDetection/
├── main.py           # Main application code
├── requirements.txt  # Python dependencies
├── .env              # User-specific config (not committed)
├── .env.sample       # Sample config template
├── yamnet_class_map.csv # Downloaded automatically for YAMNet labels
```

---

## Notes
- The `.env` file is ignored by git to protect your secrets.
- The application requires a working RTSP stream with audio.
- If you see errors about missing modules, install them with `pip install <modulename>` in your venv.
- If you see errors about ffmpeg, make sure it is installed and on your PATH.

---

## Troubleshooting
- **No module named 'cv2', 'ffmpeg', 'dotenv', etc.**
  - Run `pip install -r requirements.txt` in your venv.
- **ffmpeg not found**
  - Download and add ffmpeg to your PATH.
- **No dog barks detected**
  - Make sure your RTSP stream has audio and the URL is correct.
  - Try increasing the volume or moving the microphone closer to the dog.

---

## License
MIT License

---

## Credits
- [YAMNet](https://tfhub.dev/google/yamnet/1) by Google
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

---

Enjoy real-time dog bark detection! 🐶🎤
