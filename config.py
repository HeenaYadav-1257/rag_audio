import os

# Paths
AUDIO_DIR = "audio_files"
WEAVIATE_URL = "http://localhost:8080"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# Audio processing
CHUNK_DURATION = 50  # Reduced to 10 seconds
CHUNK_OVERLAP = 5  # seconds
SAMPLE_RATE = 16000

# Models
WHISPER_MODEL = "small"  # Switching to tiny for testing
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
MODEL_PATH = "models/mistral/mistral-7b-instruct-v0.1.Q4_0.gguf"

# Gradio
GRADIO_PORT = 7860