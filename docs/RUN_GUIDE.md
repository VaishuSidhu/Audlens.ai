# AudLens Execution Guide

To run the complete AudLens AI application (Frontend + Backend), follow these steps. You will need **two separate terminal windows** open.

---

## 1. Start the Backend (AI Engine)
The backend handles the audio processing and deepfake detection using TensorFlow.

### Steps:
1.  **Open Terminal 1**.
2.  **Navigate to the root directory**: `cd audio-guard-ai`
3.  **Activate Virtual Environment** (Optional but Recommended):
    ```powershell
    # On Windows:
    .\venv\Scripts\activate
    ```
4.  **Install Dependencies** (If not done):
    ```powershell
    pip install -r backend/requirements.txt
    ```
5.  **Run the Backend**:
    ```powershell
    python backend/app.py
    ```
    *Wait until you see `Uvicorn running on http://0.0.0.0:8000`.*

---

## 2. Start the Frontend (Web Interface)
The frontend provides the user interface for uploading files and viewing forensic results.

### Steps:
1.  **Open Terminal 2**.
2.  **Navigate to the root directory**: `cd audio-guard-ai`
3.  **Install Dependencies** (If not done):
    ```bash
    npm install  # or bun install
    ```
4.  **Run the Frontend**:
    ```bash
    npm run dev  # or bun dev
    ```
5.  **Open the App**: Click the link shown in the terminal (usually `http://localhost:5173`).

---

## Troubleshooting
- **Port 8000 Error**: If the backend says "Port 8000 already in use", make sure you don't have another instance of the backend running.
- **Analysis Fails**: Ensure the backend is running *before* you click "Analyze" in the frontend.
- **FFmpeg missing**: If you get a conversion error, ensure FFmpeg is installed on your system.
