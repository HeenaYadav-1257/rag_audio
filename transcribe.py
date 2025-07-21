import os
import logging
import numpy as np
import contextlib
import warnings
import whisper
from config import WHISPER_MODEL
from pydub import AudioSegment
from config import SAMPLE_RATE
from tempfile import NamedTemporaryFile


# 🔕 Suppress specific warnings (optional)
warnings.filterwarnings("ignore", category=ResourceWarning)

# 🛠 Setup logger to only show INFO and ERROR
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

model = None

def export_chunk(chunk, temp_file):
    logger.info(f"Exporting chunk to temp file: {temp_file.name}")
    with contextlib.closing(temp_file):
        chunk.set_frame_rate(SAMPLE_RATE).export(temp_file.name, format="wav")
    return temp_file.name

def transcribe_audio(audio_path, chunk):
    global model  # ✅ ADD THIS LINE
    if model is None:
        print(f"[🔊 Loading Whisper model: {WHISPER_MODEL}]")
        model = whisper.load_model(WHISPER_MODEL, device="cpu")


    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        wav_path = export_chunk(chunk, temp_file)
        try:
            logger.info(f"Transcribing chunk from {wav_path}")
            result = whisper.transcribe(model, wav_path, fp16=False)
            return result["text"]
        except Exception as e:
            logger.error(f"Error while transcribing: {e}")
            return f"[ERROR] {e}"
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
                logger.debug(f"Deleted temporary file: {wav_path}")

def process_chunks(audio_chunks):
    transcriptions = []
    for i, chunk in enumerate(audio_chunks):
        try:
            transcription = transcribe_audio(f"chunk_{i}", chunk)
            transcriptions.append(transcription)
        except Exception as e:
            logger.error(f"Transcription failed for chunk {i}: {e}")
            transcriptions.append(f"Transcription failed for chunk {i}: {e}")
    return " ".join([t.strip().replace("\n", " ") for t in transcriptions])

