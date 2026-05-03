import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import model.config as config
import matplotlib.pyplot as plt
import os

def save_spectrogram(log_mel_spec, save_path):
    """Saves a high-quality spectrogram image for forensic reporting."""
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel_spec, aspect='auto', origin='lower', cmap='magma')
    plt.title("Deep Forensic Spectrogram Analysis")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def augment_audio(y, sr):
    """
    Applies explicit data augmentation: Noise injection, Speed change, and Pitch shifting.
    """
    # 1. Noise Injection
    if np.random.random() > 0.5:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape)
    
    # 2. Speed Change
    if np.random.random() > 0.5:
        speed_rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=speed_rate)
        
    # 3. Pitch Shifting
    if np.random.random() > 0.5:
        n_steps = np.random.randint(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        
    return y

# Load pre-trained models once
PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
WAV2VEC_MODEL.eval()

def extract_dual_features(file_path, augment=False):
    """
    Extracts both Spectral (Mel-spec) and Semantic (Wav2Vec) features.
    Fusion of these allows detecting both 'robotic' artifacts and 'vocal' mismatches.
    """
    try:
        # 1. Load and standardise audio
        y, sr = librosa.load(file_path, sr=16000)
        
        if augment:
            y = augment_audio(y, sr)
            
        target_length = 16000 * 3
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
            
        # --- FEATURE A: Log-Mel Spectrogram (Spatial Artifacts) ---
        mel_spec = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=config.N_MELS)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-6)
        spectral_feat = log_mel_spec[..., np.newaxis] # (128, 94, 1)
        
        # --- FEATURE B: Wav2Vec 2.0 (Semantic/Humanity Features) ---
        inputs = PROCESSOR(y, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = WAV2VEC_MODEL(**inputs)
        semantic_feat = outputs.last_hidden_state.squeeze(0).numpy() # (time_steps, 768)
        
        return spectral_feat, semantic_feat
    except Exception as e:
        print(f"Error extracting dual features from {file_path}: {e}")
        return None, None
