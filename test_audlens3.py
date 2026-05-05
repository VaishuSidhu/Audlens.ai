import sys
import os
sys.path.append(r'c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/backend')
from predict import predict
import librosa
import numpy as np

filepath = r'c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/data/real/audlens-aud-3.ogg'
res = predict(filepath)

print("Final Prediction:", res["prediction"])
