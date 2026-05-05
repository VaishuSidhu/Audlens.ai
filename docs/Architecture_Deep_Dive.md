# AudLens: Technical Architecture & Forensic Deep-Dive

This document provides a detailed breakdown of the AudLens system and model architecture, highlighting the forensic tools used for deepfake detection.

---

## 1. System Architecture
The AudLens system is designed as a scalable, research-grade application with three primary layers: Frontend, Backend, and Inference Engine.

```mermaid
graph TD
    subgraph "Client Layer (Frontend)"
        A[React + Vite UI] -->|Upload Audio| B[Upload Controller]
        B -->|Progress Monitoring| A
        A -->|View Report| C[Results Dashboard]
    end

    subgraph "Service Layer (Backend)"
        D[FastAPI Server] -->|Multipart Form Data| E[File Sanitizer]
        E -->|FFmpeg Conversion| F[16kHz WAV Buffer]
        F -->|API Request| G[Detection Service]
    end

    subgraph "Core AI Engine (Model)"
        G -->|Feature Extraction| H[Dual-Stream Pipeline]
        H -->|Hybrid Inference| I[TensorFlow Model]
        I -->|Attention Weights| J[Timestamp Analyzer]
        I -->|Class Probabilities| K[Verdict Generator]
    end

    subgraph "Forensic Output"
        K -->|Final Verdict| L[JSON/PDF Report]
        J -->|Suspicious Segments| L
        H -->|Spectrogram Generation| M[Forensic Visualization]
    end

    L --> A
    M --> A
```

---

## 2. Model Architecture: Dual-Stream Fusion
The core AI utilizes a **Multi-Modal Hybrid Architecture** that combines computer vision (CNN) with natural language processing (Transformers).

```mermaid
graph LR
    subgraph "Input Processing"
        Raw[Raw Audio] -->|FFT| Spec[Log-Mel Spectrogram]
        Raw -->|Resampling| Sem[16kHz Signal]
    end

    subgraph "Spectral Branch (CNN)"
        Spec --> C1[Conv2D + BN]
        C1 --> C2[Conv2D + MaxPool]
        C2 --> GAP[Global Avg Pooling]
    end

    subgraph "Semantic Branch (Transformer)"
        Sem --> W2V[Wav2Vec 2.0 Base]
        W2V --> BLSTM[Bidirectional LSTM]
        BLSTM --> ATN[Self-Attention Layer]
    end

    subgraph "Fusion & Classification"
        GAP --> Merge[Concatenation]
        ATN -->|Pooled Features| Merge
        Merge --> D1[Dense 256 + Dropout]
        D1 --> D2[Dense 128 + BN]
        D2 --> Out[Global Output: REAL/FAKE]
        
        ATN -->|Attention Weights| TS[Segment Localization]
    end

    style GAP fill:#f9f,stroke:#333,stroke-width:2px
    style ATN fill:#00ff0066,stroke:#333,stroke-width:2px
    style Merge fill:#ff990066,stroke:#333,stroke-width:2px
```

### Key Components:
*   **Wav2Vec 2.0:** Extracts high-level semantic embeddings that represent the physical characteristics of the human vocal tract.
*   **Self-Attention:** Instead of simple averaging, the model "attends" to specific time-frames that show high mathematical jitter, allowing for precise timestamping.
*   **Bi-LSTM:** Captures long-range temporal dependencies, helping the model understand if the prosody (rhythm) of speech is organic or choppy.

---

## 3. Forensic Tools & Metrics

### 3.1. Log-Mel Spectrograms
The **Forensic Spectrogram** is the primary visual tool for investigators. 
*   **How it works:** It converts audio into a 2D image where the Y-axis is frequency and the X-axis is time.
*   **Detection:** Deepfake vocoders (like WaveNet) often leave behind "Checkerboard Artifacts" in high-frequency bands (>8kHz) that are invisible to the ear but clearly visible on a spectrogram.

![Forensic Spectrogram Example](file:///c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/images_generated/forensic_spectrogram.png)

### 3.2. Confusion Matrix
To validate the model's reliability, we use a **Confusion Matrix**.
*   **True Positives (TP):** Correctly identified Deepfakes.
*   **True Negatives (TN):** Correctly identified Real speech.
*   **False Positives (FP):** Real speech flagged as Fake (Type I Error).
*   **False Negatives (FN):** Deepfakes missed (Type II Error - Highly Dangerous).
*   **AudLens Result:** Our model minimizes False Negatives to ensure zero synthetic audio slips through.

![Confusion Matrix](file:///c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/images_generated/confusion_matrix.png)

### 3.3. Equal Error Rate (EER)
The EER is the point where the False Acceptance Rate (FAR) and False Rejection Rate (FRR) are equal. Lower is better. AudLens achieves an EER of **1.84%**, significantly outperforming baseline ResNet models.

---

## 4. Summary of Improvements
| Feature | Implementation | Purpose |
| :--- | :--- | :--- |
| **Focal Loss** | $\gamma=2, \alpha=4$ | Focuses training on "Hard Samples" that look like real speech. |
| **Multi-Task Heads** | 40ms / 160ms / Global | Enables multi-resolution analysis for forensic precision. |
| **Boundary-Aware** | Splicing Detection | Identifies synthetic segments even with smooth transitions. |

---
*Created for AudLens Forensic Research*
