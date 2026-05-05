import os
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import soundfile as sf
from model_loader import ModelLoader
from model.utils.audio_processing import extract_dual_features, save_spectrogram
import model.config as config


# ─────────────────────────────────────────────────────────────
# ACOUSTIC FORENSIC HELPERS
# ─────────────────────────────────────────────────────────────

def compute_phase_irregularity(y, sr):
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    phase = np.angle(D)
    phase_diff = np.diff(phase, axis=1)
    smoothness = np.mean(np.abs(np.diff(phase_diff, axis=1)))
    score = 1.0 - min(smoothness / 1.5, 1.0)
    return round(float(score), 4)


def compute_jitter_shimmer(y, sr):
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
    voiced = f0[f0 > 0]
    if len(voiced) < 10:
        return 0.5, 0.5
    periods = 1.0 / voiced
    jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    shimmer = float(np.std(rms) / (np.mean(rms) + 1e-9))
    jitter_suspicious = 1.0 - min(jitter / 0.010, 1.0) if jitter < 0.010 else min((jitter - 0.010) / 0.020, 1.0)
    shimmer_suspicious = max(0.0, 1.0 - shimmer / 0.05) if shimmer < 0.05 else 0.0
    return round(jitter_suspicious, 4), round(shimmer_suspicious, 4)


def compute_pause_uniformity(y, sr):
    intervals = librosa.effects.split(y, top_db=35)
    if len(intervals) < 3: return 0.0
    gaps = []
    for i in range(1, len(intervals)):
        gaps.append((intervals[i][0] - intervals[i - 1][1]) / sr)
    if len(gaps) < 2: return 0.0
    cov = np.var(gaps) / (np.mean(gaps) + 1e-9)
    return round(float(max(0.0, 1.0 - cov / 0.5)), 4)


def acoustic_verdict(phase_score, jitter_score, shimmer_score, pause_score):
    composite = (phase_score * 0.35 + jitter_score * 0.25 + shimmer_score * 0.20 + pause_score * 0.20)
    if composite >= 0.65: return "FAKE", composite
    if composite >= 0.40: return "SUSPICIOUS", composite
    return "REAL", composite


# ─────────────────────────────────────────────────────────────
# MAIN PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────

def predict(filepath: str):
    model = ModelLoader.get_model()
    y_full, sr = librosa.load(filepath, sr=16000)
    total_duration = librosa.get_duration(y=y_full, sr=sr)

    # Global forensics
    phase_g = compute_phase_irregularity(y_full, sr)
    jit_g, shim_g = compute_jitter_shimmer(y_full, sr)
    pause_g = compute_pause_uniformity(y_full, sr)
    acoustic_label_g, acoustic_score_g = acoustic_verdict(phase_g, jit_g, shim_g, pause_g)

    # Sliding Window
    window_size = 3.0
    step_size = 1.5
    segments = []

    for start in np.arange(0, max(total_duration - window_size + 0.1, window_size), step_size):
        start_sample = int(start * sr)
        end_sample = int((start + window_size) * sr)
        y_chunk = y_full[start_sample:end_sample]
        if len(y_chunk) < int(window_size * sr):
            y_chunk = np.pad(y_chunk, (0, int(window_size * sr) - len(y_chunk)))

        fd, tmp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        try:
            sf.write(tmp_path, y_chunk, sr)
            spec, sem = extract_dual_features(tmp_path)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

        if spec is None: continue
        X = [np.expand_dims(spec, axis=0), np.expand_dims(sem, axis=0)]
        preds = model.predict(X, verbose=0)
        
        if isinstance(preds, list):
            prob = preds[0][0]
            attn_spike = len(preds) > 1 and np.max(preds[1][0]) > (np.mean(preds[1][0]) * 3.5)
        else:
            prob = preds[0]
            attn_spike = False

        fake_prob = float(prob[0])
        
        # INCREASED SENSITIVITY
        if fake_prob > 0.30: # High trace = FAKE
            final_v = "FAKE"
            reason = "Synthetic spectral artifacts"
        elif fake_prob > 0.10: # Minor trace = SUSPICIOUS
            final_v = "SUSPICIOUS"
            reason = "Atypical prosody variance"
        else:
            final_v = "REAL"
            reason = "Natural vocal consistency"

        if final_v == "REAL" and attn_spike:
            final_v = "SUSPICIOUS"
            reason = "Micro-resolution artifact spike"

        segments.append({
            "start": round(float(start), 2),
            "end": round(float(start + window_size), 2),
            "prediction": final_v,
            "confidence": round(fake_prob if final_v != "REAL" else (1.0 - fake_prob), 2),
            "attn_spike": bool(attn_spike)
        })

    # Global logic
    centroids = librosa.feature.spectral_centroid(y=y_full, sr=sr)[0]
    freq_var = float(np.var(centroids))
    is_human = freq_var > 450000

    f_count = sum(1 for s in segments if s["prediction"] == "FAKE")
    s_count = sum(1 for s in segments if s["prediction"] == "SUSPICIOUS")
    total = len(segments)
    f_ratio = f_count / total if total > 0 else 0
    nr_ratio = (f_count + s_count) / total if total > 0 else 0

    if is_human:
        # HUMAN BRANCH: Forgive transient noise
        if f_ratio >= 0.50: global_prediction = "FAKE"
        elif f_ratio >= 0.10 or f_count >= 1: global_prediction = "HYBRID"
        elif nr_ratio > 0.15: global_prediction = "SUSPICIOUS"
        else: global_prediction = "REAL"
    else:
        # AI BRANCH: Extreme sensitivity
        if f_ratio >= 0.05 or f_count >= 1: global_prediction = "FAKE"
        elif nr_ratio > 0.05: global_prediction = "HYBRID"
        else: global_prediction = "SUSPICIOUS"

    # Global Acoustic Override
    if global_prediction == "REAL" and acoustic_label_g != "REAL":
        global_prediction = "SUSPICIOUS"

    if total == 1: global_prediction = segments[0]["prediction"]
    
    # UI alignment
    if global_prediction == "FAKE":
        for s in segments:
            if s["prediction"] == "REAL": s["prediction"] = "FAKE"

    spec_img, _ = extract_dual_features(filepath)
    image_name = f"forensic_{os.path.basename(filepath)}.png"
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'images_generated'))
    os.makedirs(image_dir, exist_ok=True)
    save_path = os.path.join(image_dir, image_name)
    save_spectrogram(spec_img.squeeze(), save_path)

    return {
        "prediction": global_prediction,
        "confidence": round(float(np.mean([s["confidence"] for s in segments])), 2) if total > 0 else 0,
        "segments": segments,
        "duration": round(total_duration, 2),
        "forensics": {
            "spectral_consistency": "LOW" if f_count > 0 else "HIGH",
            "detected_fake_points": [s["start"] for s in segments if s["prediction"] == "FAKE"],
            "human_confidence": round(1.0 - f_ratio, 2),
            "cnn_score": round(f_ratio, 2),
            "phase_irregularity": phase_g,
            "jitter_score": jit_g,
            "shimmer_score": shim_g,
            "pause_uniformity": pause_g,
            "acoustic_verdict": acoustic_label_g,
            "acoustic_composite": acoustic_score_g
        },
        "spectrogram_url": f"/static/{image_name}",
        "spectrogram_path": save_path
    }
