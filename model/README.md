# Deepfake Voice Detection

This project uses a **Convolutional Neural Network (CNN)** to detect deepfake voices by analyzing their waveform spectrograms. It can classify audio recordings as either **REAL** or **FAKE** with high confidence.

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
Install all required libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start: Run a Detection

You can analyze any `.wav` file directly using the `detect.py` script. This script automatically handles the conversion to a spectrogram and runs the model.

### Test with the included demo:
```bash
python detect.py audio_fake/demo.wav
```

### Analyze your own file:
```bash
python detect.py path/to/your_voice.wav
```

---

## 🛠️ Full Pipeline (Training & Customization)

If you want to train the model on your own dataset, follow these steps:

### Step 1: Prepare Your Audio Data
Place your raw `.wav` files into the following directories:
- `audio_real/` : Put your authentic voice recordings here.
- `audio_fake/` : Put your deepfake/synthetic voice recordings here.

### Step 2: Convert Audio to Spectrograms
Run the processing script to generate the visual waveforms used by the CNN:
```bash
python process_audio.py
```
This will populate the `real/` and `fake/` folders with standardized `.png` images.

### Step 3: Train the Model
Once you have generated enough images (recommended: 100+ per category), run the training script:
```bash
python train.py
```
This will train the architecture and update `deepfake_model.h5`.

---

## 📁 Project Structure

```text
Deepfake_voice_detection/
├── deepfake_model.h5     # Pre-trained CNN weights
├── detect.py             # Main CLI tool for .wav file detection
├── predict.py            # Script for image-based prediction
├── process_audio.py      # Audio -> Spectrogram converter
├── train.py              # Model training script
├── requirements.txt      # Dependency list
│
├── audio_real/           # Source folder for authentic audio
├── audio_fake/           # Source folder for deepfake audio
├── real/                 # Generated spectrograms (Real)
└── fake/                 # Generated spectrograms (Fake)
```

---

## 🧠 Model Details

- **Input**: Waveform spectrogram image (640 × 480 RGB)
- **Architecture**: multi-layer CNN (Conv2D → MaxPool → Flatten → Dense)
- **Output**: Binary classification (Fake / Real)
- **Accuracy**: Optimized for clarity in waveform patterns and noise consistency.

---

## 🔍 File Explanations

### 🐍 Python Scripts
*   **`detect.py`**: The primary tool for users. It takes a `.wav` file, converts it to a temporary spectrogram, and predicts if it's REAL or FAKE.
*   **`process_audio.py`**: A preprocessing script that converts batches of audio files from `audio_real/` and `audio_fake/` into images for the model to study.
*   **`train.py`**: The training engine. It uses the images in the `real/` and `fake/` folders to teach the CNN how to distinguish between voice types.
*   **`predict.py`**: A testing script used to classify a single, pre-existing spectrogram image.

### 🧠 Model & Core Files

*   **`deepfake_model.h5`**: The saved brain of the AI. It contains the trained neural network weights.
*   **`requirements.txt`**: A list of all necessary Python libraries needed to run this project.

### 📁 Folders & Data

*   **`audio_real/` & `audio_fake/`**: Your "Input" folders. Place your raw `.wav` recordings here.
*   **`real/` & `fake/`**: Your "Training Data" folders. These contain the visual spectrograms generated from the audio files.
*   **`audio_fake/demo.wav`**: A pre-included sample file for immediate testing.
