import sys, os
sys.path.append(r'c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/backend')
from predict import predict

files = [
    r'c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/data/audio_fake/audlens-aud-modi.ogg',
    r'c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/data/audio_fake/audlens-aud-hu-ai.ogg',
    r'c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/data/audio_fake/audlens-aud-ai.ogg',
    r'c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/data/audio_real/audlens-aud-1.ogg',
    r'c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/data/real/real_aud-2.ogg',
    r'c:/Users/svvai/Downloads/Audlens.ai/audio-guard-ai/model/data/real/real-aud-3.ogg',
]

for fp in files:
    name = os.path.basename(fp)
    if not os.path.isfile(fp):
        print(f"[NOT FOUND] {name}")
        continue
    try:
        res = predict(fp)
        f = res.get('forensics', {})
        print(f"{name} -> {res.get('prediction')} | acoustic={f.get('acoustic_verdict')} composite={f.get('acoustic_composite')}")
    except Exception as e:
        import traceback
        print(f"[ERROR] {name}: {e}")
        traceback.print_exc()
