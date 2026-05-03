import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data') 
REAL_AUDIO_DIR = os.path.join(DATA_DIR, 'audio_real')
FAKE_AUDIO_DIR = os.path.join(DATA_DIR, 'audio_fake')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'deepfake_hybrid_v2.h5')
IMAGES_DIR = os.path.join(BASE_DIR, 'images_generated')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports_generated')

# Audio Processing Parameters
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
TIME_STEPS = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Augmentation Flags
USE_AUGMENTATION = True
