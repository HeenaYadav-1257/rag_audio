import os
from pydub import AudioSegment
import numpy as np
from config import AUDIO_DIR, CHUNK_DURATION, CHUNK_OVERLAP, SAMPLE_RATE

def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
    audio = audio.normalize()
    return audio

def chunk_audio(audio):
    chunk_ms = CHUNK_DURATION * 1000
    overlap_ms = CHUNK_OVERLAP * 1000
    chunks = []
    for start in range(0, len(audio), chunk_ms - overlap_ms):
        chunk = audio[start:start + chunk_ms]
        if len(chunk) >= chunk_ms // 2:  # Ignore short chunks
            chunks.append(chunk)
    return chunks

def get_audio_files():
    supported_formats = [".mp3", ".wav"]
    return [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if os.path.splitext(f)[1].lower() in supported_formats]