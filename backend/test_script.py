import sys
import os

# Add current directory to path so we can import predict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import predict

# Path to the specific file
test_file = r"C:\Users\svvai\Downloads\Audlens.ai\audio-guard-ai\model\Deep-Fake-Voice-Detection-main\audlens.ai-model\Deep-Fake-Voice-Detection-main\Deep-Fake-Voice-Detection-main\audio_real\real_0.wav"

if not os.path.exists(test_file):
    print(f"Error: File not found at {test_file}")
    sys.exit(1)

print(f"Testing file: {os.path.basename(test_file)}")
print("-" * 30)

try:
    result = predict(test_file)
    print("\n--- RESULTS ---")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100}%")
    
    if 'forensics' in result:
        print("\n--- FORENSIC INSIGHTS ---")
        print(f"Spectral Consistency: {result['forensics']['spectral_consistency']}")
        print(f"Neural Model Score: {result['forensics']['cnn_score']}")
        print(f"Frequency Forensic Score: {result['forensics']['frequency_score']}")
    
    if result['segments']:
        print("\n--- SUSPICIOUS SEGMENTS ---")
        for seg in result['segments']:
            print(f"  {seg['start']}s - {seg['end']}s (Conf: {seg['confidence']})")
    else:
        print("\nNo suspicious segments detected.")

except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()
