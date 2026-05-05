import os
import sys
import tempfile
import uuid

# Add parent directory to path so 'model' package can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.background import BackgroundTasks

from model_loader import ModelLoader
from predict import predict
from utils.report_generator import generate_pdf_report
import librosa
import soundfile as sf

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AudLens Backend", description="Audio Deepfake Detection System")

# Serve generated forensic images
IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'images_generated'))
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=IMAGES_DIR), name="static")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SERVE FRONTEND ---
# This serves the built React/Vite app from the dist/client folder
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dist', 'client'))
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.ogg'}

@app.on_event("startup")
async def startup_event():
    # Load model once at server startup
    ModelLoader.get_model()

def validate_file_extension(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
    return ext

def convert_to_wav(input_path: str):
    """Converts non-wav files to wav format for consistency."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.wav':
        return input_path
        
    wav_path = os.path.splitext(input_path)[0] + "_converted.wav"
    try:
        y, sr = librosa.load(input_path, sr=None)
        sf.write(wav_path, y, sr)
        return wav_path
    except Exception as e:
        print(f"Conversion error: {e}")
        return input_path # Fallback to original if conversion fails

def remove_temp_file(filepath: str):
    """Background task to remove temporary files."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error deleting temp file {filepath}: {e}")

@app.post("/analyze-audio")
async def analyze_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    ext = validate_file_extension(file.filename)
    
    temp_audio_path = ""
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
            
        # Convert to WAV if necessary
        processing_path = convert_to_wav(temp_audio_path)
        if processing_path != temp_audio_path:
            background_tasks.add_task(remove_temp_file, processing_path)

        # Run prediction
        result = predict(processing_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
        
    finally:
        # Schedule cleanup
        if temp_audio_path:
            background_tasks.add_task(remove_temp_file, temp_audio_path)


@app.post("/download-report")
async def download_report(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    ext = validate_file_extension(file.filename)
    
    temp_audio_path = ""
    temp_pdf_path = ""
    try:
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
            
        # Convert to WAV if necessary
        processing_path = convert_to_wav(temp_audio_path)
        if processing_path != temp_audio_path:
             background_tasks.add_task(remove_temp_file, processing_path)

        # Run prediction
        result = predict(processing_path)
        
        # Generate PDF report
        temp_pdf_fd, temp_pdf_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_pdf_fd) # Close file descriptor, reportlab will write to the path
        
        generate_pdf_report(result, temp_pdf_path)
        
        # Return downloadable PDF file
        # The background task will delete the audio and PDF after response is sent
        response = FileResponse(
            path=temp_pdf_path, 
            media_type='application/pdf', 
            filename=f"AudLens_Report_{uuid.uuid4().hex[:8]}.pdf"
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")
        
    finally:
        # Schedule cleanup for audio and pdf
        if temp_audio_path:
            background_tasks.add_task(remove_temp_file, temp_audio_path)
        if temp_pdf_path:
            background_tasks.add_task(remove_temp_file, temp_pdf_path)

if __name__ == "__main__":
    import uvicorn
    # Only reload on backend changes, not the whole project
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["backend"])
