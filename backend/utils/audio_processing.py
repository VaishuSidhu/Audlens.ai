import librosa
import warnings

def process_audio(filepath: str):
    """
    Loads and processes an audio file using librosa.
    Returns the audio time series and sampling rate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Load audio file (supports both mp3 and wav via librosa/audioread)
        y, sr = librosa.load(filepath, sr=None)
    
    # In a real model, you would extract required features here.
    # For example:
    # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    return y, sr
