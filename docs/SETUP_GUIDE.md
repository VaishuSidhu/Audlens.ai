# AudLens AI Setup & Installation Guide

This guide will walk you through the complete setup of the AudLens Audio Deepfake Detection system.

---

## 1. System Requirements
- **OS**: Windows 10/11, macOS, or Linux.
- **Python**: 3.9 or higher.
- **Node.js**: 18.0 or higher.
- **Hardware**: Minimum 8GB RAM (16GB recommended for TensorFlow processing).
- **FFmpeg**: Required for audio format conversion.

---

## 2. Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/VaishuSidhu/Audlens.ai.git
cd audio-guard-ai
```

### Step 2: Backend & AI Engine Setup
1.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```
2.  **Activate the Environment**:
    - **Windows**: `.\venv\Scripts\activate`
    - **Mac/Linux**: `source venv/bin/activate`
3.  **Install Python Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    ```

### Step 3: Frontend Setup
1.  **Install Node Dependencies**:
    ```bash
    npm install
    # OR if you use bun:
    bun install
    ```

### Step 4: Install FFmpeg
The backend uses FFmpeg to convert `.mp3` and `.ogg` files to `.wav`.
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your PATH.
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

---

## 3. Running the Application

You must run the **Backend** and **Frontend** simultaneously in two separate terminals.

### Terminal 1: Backend
```bash
python backend/app.py
```
Wait for `Model loaded successfully`.

### Terminal 2: Frontend
```bash
npm run dev
```
Open `http://localhost:8080` (or the port shown in your terminal).

---

## 4. Troubleshooting
- **TensorFlow Errors**: Ensure you are using Python 3.9-3.11. Python 3.12+ might have compatibility issues with older TensorFlow versions.
- **Missing Model**: If you get a "File Not Found" error for `.h5`, ensure you have pulled the latest changes from Git, as the model weights are included in the repo.
- **CORS Errors**: The backend is configured to allow `localhost`. Ensure your browser isn't blocking local requests.
