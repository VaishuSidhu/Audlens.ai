# Forensic Analysis Guide: Audio Deepfake Detection

This document outlines the forensic methodology used by the **AudLens.ai** system to distinguish between authentic human speech and AI-generated (Deepfake) audio.

## 1. The Dual-Stream Forensic Logic
Our architecture employs two distinct analysis "streams" to ensure high-accuracy detection.

```mermaid
graph TD
    A[Input Audio File] --> B[Dual-Stream Processor]
    
    B --> C[Stream 1: Spectral Branch - CNN]
    C --> C1[Detects 'Checkerboard' Artifacts]
    C --> C2[Analyzes High-Frequency Jitter]
    C --> C3[Identifies Upsampling Noise]
    
    B --> D[Stream 2: Semantic Branch - Wav2Vec 2.0]
    D --> D1[Extracts Human Vocal Tract Embeddings]
    D --> D2[Analyzes Prosody & Emotion Flow]
    D --> D3[Checks for Physiological Consistency]
    
    C1 & C2 & C3 & D1 & D2 & D3 --> E[Fusion Layer]
    E --> F[Final Forensic Verdict]
    F --> G[REAL: Bio-Signatures Verified]
    F --> H[FAKE: Synthetic Artifacts Detected]
```

## 2. Forensic Indicator Comparison

| Feature | Authentic Human Speech | AI-Generated (Deepfake) |
| :--- | :--- | :--- |
| **Pitch Stability** | Natural "Micro-Jitter" (Imperfect) | Mathematically stable or robotic |
| **Harmonic Flow** | Organic, follows vocal tract physics | May have sharp "breaks" or overlaps |
| **Spectral Noise** | Natural background/room ambiance | Digital "checkerboard" or grid artifacts |
| **Breathing** | Integrated with speech rhythm | Often missing or mechanically placed |
| **High Frequency** | Smooth roll-off | Sudden spikes or "metallic" hiss |

## 3. Why Fakes Are Caught
AI models generate audio using **Neural Upsampling**. This process inevitably leaves behind tiny, periodic mathematical patterns. While these are often inaudible to humans, they appear as distinct "pixels" or "grids" in the spectrogram. Our CNN branch is specifically trained to find these **Synthetic Fingerprints**.

---
*Created for AudLens.ai Research Project*
