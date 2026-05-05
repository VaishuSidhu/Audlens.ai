# AudLens Audio AI: End-to-End System Workflow

This document explains the complete journey of the AudLens system, from the initial model development to the live integration with the backend and frontend.

---

## 1. Phase 1: Model Development (The "Brain")
The model was built using a **Research-First** approach to ensure forensic-grade accuracy.

### A. Data Preparation
- **Dataset**: Trained on the **ASVspoof 2019** Logical Access (LA) dataset, which contains thousands of authentic human recordings and AI-generated spoofs (using various TTS and Voice Conversion algorithms).
- **Preprocessing**: All audio is normalized to **16kHz Mono WAV** format to ensure the model sees consistent data, regardless of the original source.

### B. Feature Engineering
Instead of one feature, we extract two:
1.  **Spectrograms (CNN)**: Captures digital artifacts in high frequencies.
2.  **Embeddings (Wav2Vec 2.0)**: Captures semantic and biological human speech traits.

### C. Training Logic
- **Focal Loss**: Used to focus training on "hard" samples that standard models often miss.
- **Early Stopping**: Prevents overfitting, ensuring the model generalizes well to new, unseen AI voices.

---

## 2. Phase 2: Backend Integration (The "Bridge")
We used **FastAPI** to connect the Python-based AI model to the web interface.

### A. API Endpoints
- `POST /analyze-audio`: Receives a multipart file upload, saves it temporarily, and triggers the detection engine.
- `POST /download-report`: Re-runs the analysis and generates a professional PDF using the **ReportLab** library.

### B. The Inference Engine (`predict.py`)
This is the most critical part of the integration. It implements a **Sliding Window Scan**:
1.  The audio is sliced into **3-second windows**.
2.  Each window is fed through the model independently.
3.  The results are aggregated to detect **Hybrid** audio (e.g., 10 seconds of human speech followed by 5 seconds of AI).

### C. Why These Integration Tools?
| Tool | Choice Rationale |
| :--- | :--- |
| **FastAPI** | High performance and natively handles asynchronous requests, allowing multiple users to analyze audio simultaneously. |
| **Uvicorn** | A lightning-fast ASGI server that serves the FastAPI application. |
| **FFmpeg** | The "Swiss Army Knife" for audio; handles conversion of MP3, OGG, and other formats into the 16kHz WAV format the model requires. |
| **ReportLab** | A robust Python toolkit for generating dynamic, forensic-style PDF reports. |

---

## 3. Phase 3: The End-to-End Flow
Here is how data moves through the system when a user clicks "Analyse":

1.  **User Action**: User uploads an audio file (e.g., `evidence.mp3`) via the React frontend.
2.  **Transmission**: The frontend sends a `multipart/form-data` request to the FastAPI backend.
3.  **Sanitization**: Backend uses **FFmpeg** to convert the file to a standard format and calculates its duration.
4.  **Forensic Scanning**:
    - The **Inference Engine** scans the file using the sliding window.
    - **Self-Attention** weights are extracted to identify suspicious timestamps.
5.  **Data Serialization**: The results (prediction, confidence, segments, spectrogram URL) are packed into a JSON object.
6.  **UI Rendering**:
    - The frontend receives the JSON.
    - **ResultCard**: Displays the global verdict (e.g., "FULL FAKE").
    - **SegmentTimeline**: Renders the visual heatmap of where the AI was detected.
7.  **Archiving (Optional)**: If requested, **ReportLab** generates a PDF containing all forensic evidence and timestamps for download.

---

## 4. Key Design Decisions

- **Stateless Backend**: The server doesn't store your audio permanently. Files are processed in memory or temp folders and deleted immediately after analysis for privacy.
- **Hybrid Detection**: Unlike other detectors that give a single "Yes/No", AudLens identifies *where* the fake parts are, which is essential for legal forensic work.
- **Real-time Feedback**: By using **Vite + React**, the UI updates instantly, showing a loading state while the "heavy lifting" happens on the GPU/CPU in the backend.

---
*Built with AudLens.ai © 2026*
