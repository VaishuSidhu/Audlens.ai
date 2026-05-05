import os
import sys
import numpy as np
import tensorflow as tf
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import model.config as config
from model.utils.audio_processing import PROCESSOR, WAV2VEC_MODEL

def extract_features_from_buffer(y, sr=16000):
    """Extracts features from a pre-loaded audio buffer."""
    # Standardise length to 3 seconds
    target_length = sr * 3
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
        
    # --- FEATURE A: Log-Mel Spectrogram ---
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-6)
    spectral_feat = log_mel_spec[..., np.newaxis]
    
    # --- FEATURE B: Wav2Vec 2.0 ---
    inputs = PROCESSOR(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = WAV2VEC_MODEL(**inputs)
    semantic_feat = outputs.last_hidden_state.squeeze(0).numpy()
    
    return spectral_feat, semantic_feat

def scan_file(audio_path):
    print(f"\n--- Forensic Investigation: {os.path.basename(audio_path)} ---")
    
    # Load Model
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    
    # Load full audio
    y_full, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(y_full) / sr
    print(f"Total Duration: {total_duration:.2f} seconds")
    
    # Scan in 3-second windows with 1.5s overlap
    window_size = 3.0
    step_size = 1.5
    
    results = []
    print("\nScanning timeline for synthetic artifacts...")
    print(f"{'Time (s)':<15} | {'Verdict':<10} | {'Confidence':<10}")
    print("-" * 40)
    
    for start in np.arange(0, total_duration - window_size + 0.1, step_size):
        start_sample = int(start * sr)
        end_sample = int((start + window_size) * sr)
        y_chunk = y_full[start_sample:end_sample]
        
        spec, sem = extract_features_from_buffer(y_chunk, sr)
        X = [np.expand_dims(spec, axis=0), np.expand_dims(sem, axis=0)]
        
        pred_probs = model.predict(X, verbose=0)
        # Handle multi-output
        if isinstance(pred_probs, list):
            pred_probs = pred_probs[0]
        if len(pred_probs.shape) > 1:
            pred_probs = pred_probs[0]
            
        class_names = ['FAKE', 'REAL']
        predicted_class = class_names[np.argmax(pred_probs)]
        confidence = np.max(pred_probs) * 100
        
        results.append((start, predicted_class, confidence))
        print(f"{start:>5.1f} - {start+window_size:>5.1f}s | {predicted_class:<10} | {confidence:>8.2f}%")

    # Final Hybrid Analysis
    fake_count = sum(1 for r in results if r[1] == 'FAKE')
    real_count = sum(1 for r in results if r[1] == 'REAL')
    
    print("\n--- HYBRID COMPOSITION SUMMARY ---")
    print(f"Detected Real Segments: {real_count}")
    print(f"Detected Fake Segments: {fake_count}")
    
    if fake_count > 0 and real_count > 0:
        print("\nFORENSIC VERDICT: HYBRID MANIPULATION DETECTED")
        print("Detailed analysis shows the audio contains both organic human speech and synthetic AI segments.")
    elif fake_count > 0:
        print("\nFORENSIC VERDICT: PURE DEEPFAKE")
    else:
        print("\nFORENSIC VERDICT: ORGANIC HUMAN SPEECH")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model/forensic_scan.py <path_to_audio>")
    else:
        scan_file(sys.argv[1])
