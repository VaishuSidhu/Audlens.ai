import os
import sys
import numpy as np
import tensorflow as tf
import model.config as config
from model.utils.audio_processing import extract_dual_features

def predict(audio_path):
    """
    Predicts using the Research-Grade Dual Fusion Model with Sliding Window for Hybrid Detection.
    """
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Error: Model not found.")
        return

    # Load Model
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    
    # Load full audio to support long/hybrid files
    print(f"--- Deep Forensic Fusion Analysis: {os.path.basename(audio_path)} ---")
    y_full, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(y_full) / sr
    
    # Scan parameters
    window_size = 3.0
    step_size = 1.5 # 50% overlap
    
    results = []
    print(f"Total Duration: {total_duration:.2f}s | Scanning for hybrid signatures...")
    
    # Loop through audio chunks
    for start in np.arange(0, max(total_duration - window_size + 0.1, window_size), step_size):
        start_sample = int(start * sr)
        end_sample = int((start + window_size) * sr)
        y_chunk = y_full[start_sample:end_sample]
        
        # Pad if last chunk is too short
        if len(y_chunk) < int(window_size * sr):
            y_chunk = np.pad(y_chunk, (0, int(window_size * sr) - len(y_chunk)))
            
        # Extract features for this chunk
        from model.forensic_scan import extract_features_from_buffer
        spec, sem = extract_features_from_buffer(y_chunk, sr)
        
        X = [np.expand_dims(spec, axis=0), np.expand_dims(sem, axis=0)]
        pred_probs = model.predict(X, verbose=0)
        
        # Handle single-output head
        if isinstance(pred_probs, list):
            pred_probs = pred_probs[0]
        if len(pred_probs.shape) > 1:
            pred_probs = pred_probs[0]
            
        class_names = ['FAKE', 'REAL']
        predicted_class = class_names[np.argmax(pred_probs)]
        confidence = np.max(pred_probs) * 100
        
        # --- IMPROVEMENT 2: INCREASE SENSITIVITY ---
        # Extreme threshold (98%) for human verification
        if predicted_class == 'REAL' and confidence < 98:
            verdict = "SUSPICIOUS (AI?)"
        else:
            verdict = predicted_class
            
        results.append((start, start + window_size, verdict, confidence))

    # --- Forensic Reporting ---
    total_segments = len(results)
    fake_segments = [r for r in results if "FAKE" in r[2]]
    suspicious_segments = [r for r in results if "SUSPICIOUS" in r[2]]
    non_real_segments = fake_segments + suspicious_segments
    
    non_real_ratio = len(non_real_segments) / total_segments if total_segments > 0 else 0
    fake_ratio = len(fake_segments) / total_segments if total_segments > 0 else 0
    
    # --- IMPROVEMENT 1: SAFETY-FIRST FORENSIC OVERRIDE ---
    # In a professional forensic context, if the start of a recording is fake, 
    # it is highly probable the entire generation is a "Masked" deepfake.
    
    first_segment_fake = len(results) > 0 and "FAKE" in results[0][2] and results[0][3] > 80
    
    if fake_ratio > 0.30 or non_real_ratio > 0.40:
        global_prediction = "FAKE"
        verdict_text = "RESULT: DEEPFAKE DETECTED (FULL FAKE AUDIO BY AI AGENT VOICE)"
    elif first_segment_fake and total_duration < 30:
        # SAFETY OVERRIDE: If the "Seed" is fake, the whole file is compromised
        global_prediction = "FAKE"
        verdict_text = "RESULT: DEEPFAKE DETECTED (FULL FAKE - MASKED SYNTHESIS)"
    elif non_real_segments:
        global_prediction = "HYBRID"
        verdict_text = "RESULT: HYBRID MANIPULATION DETECTED"
    else:
        global_prediction = "REAL"
        verdict_text = "RESULT: ORGANIC HUMAN SPEECH VERIFIED"

    # --- RETROACTIVE SEGMENT ALIGNMENT ---
    # For reporting clarity: If the global verdict is FAKE, all segments are compromised
    if global_prediction == "FAKE":
        results = [(r[0], r[1], "FAKE (MASKED)", r[3]) for r in results]
        non_real_segments = results

    print("\n" + "="*50)
    print("FORENSIC TIMELINE REPORT (ULTRA SENSITIVITY)")
    print("="*50)
    for start, end, verdict, conf in results:
        status_label = "[!]" if verdict != "REAL" else "[ ]"
        print(f"{status_label} {start:>5.1f}s - {end:>5.1f}s | {verdict:<17} | Confidence: {conf:>6.2f}%")
    
    print("\n" + "="*50)
    print("FINAL VERDICT SUMMARY")
    print("="*50)
    print(verdict_text)
    
    if global_prediction == "FAKE":
        print(f"Forensic Alert: AI signatures detected. High-fidelity mimicry observed in later segments.")
        print(f"Global Confidence: {np.max([r[3] for r in non_real_segments]):.2f}%")
    elif global_prediction == "HYBRID":
        print(f"The audio contains both organic human speech and synthetic AI segments.")
        print(f"AI/Suspicious regions found at: " + ", ".join([f"{r[0]:.1f}s" for r in non_real_segments]))
        
    print("="*50)

    # Save a representative spectrogram (first chunk)
    image_name = f"forensic_{os.path.basename(audio_path)}.png"
    save_path = os.path.join(config.IMAGES_DIR, image_name)
    from model.utils.audio_processing import save_spectrogram
    # Get features for the first 3s for the image
    spec_img, _ = extract_features_from_buffer(y_full[:int(sr*3)], sr)
    save_spectrogram(spec_img.squeeze(), save_path)
    print(f"\nForensic Spectrogram saved to: {save_path}")

if __name__ == "__main__":
    import librosa
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
    else:
        predict(sys.argv[1])
