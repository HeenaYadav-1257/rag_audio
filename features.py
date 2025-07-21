import librosa
import numpy as np
from io import BytesIO
from config import SAMPLE_RATE

def extract_features(audio_chunk):
    # Convert pydub AudioSegment to numpy array
    samples = np.array(audio_chunk.get_array_of_samples(), dtype=np.float32)
    samples = samples / 32768.0  # Normalize to [-1, 1]
    mfcc = librosa.feature.mfcc(y=samples, sr=SAMPLE_RATE, n_mfcc=13)
    return mfcc