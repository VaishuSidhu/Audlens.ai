import librosa
import numpy as np

def test_heuristics(filepath):
    y, sr = librosa.load(filepath, sr=16000)
    
    # Centroid variance
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_var = np.var(cent)
    
    # Gap variance
    intervals = librosa.effects.split(y, top_db=30)
    gaps = []
    for i in range(1, len(intervals)):
        gaps.append(intervals[i][0] - intervals[i-1][1])
    gap_var = np.var(gaps) if len(gaps) > 1 else 0
    
    # F0 variance
    f0 = librosa.yin(y, fmin=50, fmax=400)
    # mask unvoiced
    rms = librosa.feature.rms(y=y)[0]
    voiced_f0 = f0[rms > np.mean(rms)]
    pitch_var = np.var(voiced_f0) if len(voiced_f0) > 0 else 0
    
    print(f"File: {filepath}")
    print(f"Centroid Var: {cent_var:.2f}")
    print(f"Gap Var:      {gap_var:.2f}")
    print(f"Pitch Var:    {pitch_var:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    test_heuristics("C:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/data/real/audlens-aud-3.ogg")
    test_heuristics("C:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/data/audio_fake/audlens-aud-modi.ogg")
