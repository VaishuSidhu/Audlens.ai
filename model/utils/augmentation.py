import librosa
import numpy as np
import random

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_stretch(y, rate=None):
    if rate is None:
        rate = random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr, steps=None):
    if steps is None:
        steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def volume_scale(y, factor=None):
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    return y * factor

def apply_augmentation(y, sr):
    """Randomly applies one or more augmentations."""
    aug_func = random.choice([add_noise, time_stretch, pitch_shift, volume_scale])
    
    if aug_func == pitch_shift:
        return aug_func(y, sr)
    else:
        return aug_func(y)
