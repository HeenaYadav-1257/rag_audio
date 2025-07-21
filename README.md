# Audio RAG System: Retrieval-Augmented Generation for Auditory Data

## Project Overview

This project implements an Audio Retrieval-Augmented Generation (RAG) system designed to augment Large Language Models (LLMs) with contextual information derived from audio files. The system transcribes audio, embeds the text, retrieves relevant segments based on user queries, and then leverages a local LLM (Mistral-7B) to provide accurate and contextually informed answers. This is crucial for making unstructured audio content queryable and reducing LLM reliance on parametric knowledge alone.

## Architecture

The Audio RAG system operates through a sequential pipeline orchestrated by `ui.py` and `run.sh`:

1.  **Audio Ingestion:** Local audio files from the `audio_files` directory are loaded.
2.  **Audio Chunking & Feature Extraction:** Loaded audio is normalized, resampled, and split into smaller chunks. Mel-frequency cepstral coefficients (MFCCs) are extracted from these chunks.
3.  **Audio Transcription (ASR):** Each audio chunk is transcribed into text using a powerful ASR model.
4.  **Embedding Generation:** The combined transcription is tokenized into sentences, which are then converted into dense vector embeddings.
5.  **Vector Database Storage:** These embeddings, along with associated metadata (transcription, file ID, chunk ID), are stored in a vector database (Weaviate).
6.  **User Query & Retrieval:** User text queries (via Gradio UI) are embedded. A similarity search is performed in Weaviate to retrieve the most relevant text chunks from the audio transcripts.
7.  **LLM Augmentation & Generation:** The retrieved text context is combined with the user's query and fed to a local LLM, which generates the final answer.
8.  **User Interface:** A Gradio application provides the interactive front-end.
9.  **Caching:** Redis is used for caching responses and intermediate data.

**Visual Representation:**
(Refer to Figure 3.2: Audio RAG System Architecture in your report for a detailed visual.)

## Features

* **Comprehensive Audio Processing:** Handles various audio formats, normalization, chunking, and MFCC feature extraction.
* **State-of-the-Art ASR:** Utilizes `Whisper-medium-v3` for high-accuracy speech-to-text transcription.
* **Robust Text Embedding:** Employs `all-mpnet-base-v2` for generating semantically rich embeddings from audio transcripts.
* **Efficient Vector Search:** Leverages `Weaviate` for fast and scalable storage and retrieval of vector embeddings.
* **Local LLM Integration:** Integrates with a local, quantized `Mistral-7B` LLM using `llama.cpp` for privacy-preserving and resource-efficient answer generation.
* **User-Friendly Interface:** Provides a user-friendly web UI built with `Gradio`.
* **Response Caching:** Implements `Redis` caching to improve response times for repeated queries.
* **Cross-Reference to Images:** Includes a supplementary feature to display images from an 'images' folder if their filenames match keywords in the generated LLM answer.

## Prerequisites

Before running the project, ensure you have the following installed:

* Python 3.9+ (compatible with listed libraries).
* Docker (for running Weaviate locally).
* FFmpeg (required by `ffmpy` for audio processing and `pydub` for certain operations).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>/audio_rag_project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv audio_rag
    source audio_rag/bin/activate  # On Windows: audio_rag\Scripts\activate
    ```

3.  **Install dependencies:**
    The `requirements.txt` for this project is generated from your provided `pip freeze` output:
    ```
    accelerate==1.8.1
aiofiles==24.1.0
aiohappyeyeballs==2.6.1
aiohttp==3.12.13
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.9.0
async-timeout==5.0.1
attrs==25.3.0
audioread==3.0.1
Authlib==1.6.0
certifi==2025.6.15
cffi==1.17.1
charset-normalizer==3.4.2
click==8.2.1
cryptography==45.0.4
datasets==3.6.0
decorator==5.2.1
deprecation==2.1.0
dill==0.3.8
diskcache==5.6.3
exceptiongroup==1.3.0
fastapi==0.115.14
ffmpy==0.6.0
filelock==3.13.1
frozenlist==1.7.0
fsspec==2024.6.1
gradio==5.35.0
gradio_client==1.10.4
groovy==0.1.2
grpcio==1.73.1
grpcio-health-checking==1.73.1
grpcio-tools==1.73.1
h11==0.16.0
hf-xet==1.1.5
httpcore==1.0.9
httpx==0.26.0
huggingface-hub==0.33.2
idna==3.10
iniconfig==2.1.0
Jinja2==3.1.6
joblib==1.5.1
lazy_loader==0.4
librosa==0.11.0
llama_cpp_python==0.3.9
llvmlite==0.44.0
lxml==6.0.0
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
more-itertools==10.7.0
mpmath==1.3.0
msgpack==1.1.1
multidict==6.6.3
multiprocess==0.70.16
networkx==3.3
nltk==3.9.1
numba==0.61.2
numpy==1.26.4
openai-whisper==20250625
orjson==3.10.18
packaging==25.0
pandas==2.3.0
pillow==11.0.0
platformdirs==4.3.8
pluggy==1.6.0
pooch==1.8.2
propcache==0.3.2
protobuf==6.31.1
psutil==7.0.0
pyarrow==20.0.0
pycparser==2.22
pydantic==2.11.7
pydantic_core==2.33.2
pydub==0.25.1
Pygments==2.19.2
PyMuPDF==1.26.3
pytest==8.4.1
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-multipart==0.0.20
pytz==2025.2
PyYAML==6.0.2
redis==6.2.0
regex==2024.11.6
requests==2.32.4
rich==14.0.0
ruff==0.12.1
safehttpx==0.1.6
safetensors==0.5.3
scikit-learn==1.7.0
scipy==1.15.3
semantic-version==2.10.0
sentence-transformers==5.0.0
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
soundfile==0.13.1
soxr==0.5.0.post1
starlette==0.46.2
sympy==1.13.3
threadpoolctl==3.6.0
tiktoken==0.9.0
tokenizers==0.21.2
tomli==2.2.1
tomlkit==0.13.3
torch==2.2.2
torchaudio==2.2.2
torchvision==0.17.2
tqdm==4.67.1
transformers==4.53.0
typer==0.16.0
typing-inspection==0.4.1
typing_extensions==4.14.1
tzdata==2025.2
urllib3==2.5.0
uvicorn==0.35.0
validators==0.22.0
weaviate-client==4.4.2
websockets==15.0.1
xxhash==3.5.0
yarl==1.20.1

    ```
    Create a `requirements.txt` file from this output in your project root and then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install FFmpeg:**
    * **macOS (using Homebrew):** `brew install ffmpeg`
    * **Linux (Debian/Ubuntu):** `sudo apt-get update && sudo apt-get install ffmpeg`
    * **Windows:** Download from the official FFmpeg website and add it to your system's PATH.

5.  **Start Weaviate (via Docker):**
    Ensure Docker is running. Then, start a local Weaviate instance. A basic `docker-compose.yml` might look like this (adjust ports/volumes as needed):
    ```yaml
    version: '3.8'
    services:
      weaviate:
        command:
        - --host
        - 0.0.0.0
        - --port
        - '8080'
        - --scheme
        - http
        image: semitechnologies/weaviate:1.24.1 # Or a newer stable version
        ports:
          - "8080:8080"
          - "50051:50051"
        volumes:
          - ./weaviate_data:/var/lib/weaviate
        restart: on-failure
        environment:
          QUERY_DEFAULTS_LIMIT: 25
          AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
          PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
          DEFAULT_VECTORIZER_MODULE: 'none' # Or 'text2vec-transformers' if using Weaviate's module
          ENABLE_MODULES: '' # e.g. 'text2vec-transformers,generative-openai'
          CLUSTER_HOSTNAME: 'node1'
    ```
    Run:
    ```bash
    docker-compose up -d
    ```

6.  **Download LLM (Mistral-7B GGUF):**
    Download the `Mistral-7B-v0.1-GGUF` model file (specifically, `mistral-7b-instruct-v0.1.Q4_0.gguf` as defined in `config.py`) and place it in the `models/mistral/` directory relative to your project root.

## Usage

1.  **Prepare your audio data:**
    Place your `.wav`, `.mp3`, etc., audio files in the designated `audio_files` directory.

2.  **Run the data ingestion/transcription script:**
    This script will transcribe audio, chunk text, generate embeddings, and populate Weaviate.
    ```bash
    bash run.sh
    ```
    This script calls `python ui.py`, which, upon launch, executes `index_all_audio()` to preprocess the entire audio dataset.

3.  **Start the Gradio UI:**
    The `run.sh` script already starts the UI.
    This will launch the Gradio interface in your web browser, typically at `http://localhost:7860`.

4.  **Interact with the system:**
    * Enter your text queries in the "Ask your question" input field.
    * Click "Get Answer".
    * The system will retrieve relevant audio transcript chunks from Weaviate and feed them to the Mistral-7B LLM.
    * The generated augmented response will be displayed in the "Answer" textbox.
    * If keywords from the answer are found in image filenames within the `images` folder, those images will be displayed in the gallery.

## Project Structure
├── config.py           # Configuration parameters (paths, model names, ports, audio settings)
├── preprocess.py       # Functions for loading and chunking audio files
├── features.py         # Function for extracting MFCC features from audio chunks
├── transcribe.py       # Handles audio transcription using Whisper
├── embed.py            # Generates SentenceTransformer embeddings for text
├── vector_store.py     # Manages Weaviate connection, schema, storage, and retrieval
├── llm.py              # Handles LLM (Llama.cpp) initialization and response generation
├── ui.py               # Gradio UI application and overall orchestration of audio processing
├── run.sh              # Shell script to start the UI (and implicitly trigger audio indexing)
├── models/             # Directory for LLM models
│   └── mistral/        # Subdirectory for Mistral models
│       └── mistral-7b-instruct-v0.1.Q4_0.gguf # Specific Mistral model
├── audio_files/        # Directory for raw audio input files
├── images/             # Directory for image files (used by fetch_relevant_images)
├── weaviate_data/      # Weaviate data volume (from docker-compose, created on run)
└── README.md


## Technologies Used (Detailed)

* **Core Deep Learning Framework:** PyTorch (`torch==2.2.2`), PyTorch Audio (`torchaudio==2.2.2`), PyTorch Vision (`torchvision==0.17.2`). These provide the foundational numerical and tensor operations for various models and data processing steps.
* **Numerical Computing:** NumPy (`numpy==1.26.4`). Used extensively for array operations in audio processing, feature extraction, and embedding handling.
* **Automatic Speech Recognition (ASR):**
    * Library: OpenAI Whisper (`openai-whisper==20250625`).
    * Model: `Whisper-medium-v3` (configured as `WHISPER_MODEL = "small"` in `config.py`).
* **Text Embedding Model (for audio transcripts & query):**
    * Library: `sentence-transformers==5.0.0`.
    * Specific Model: `all-mpnet-base-v2`.
* **Large Language Model (LLM):**
    * Model: `Mistral-7B-v0.1` (instruction-tuned variant).
    * Format/Quantization: `Q4_0.gguf`.
* **LLM Inference Backend:**
    * `llama_cpp_python==0.3.9`.
* **LLM Ecosystem Libraries:**
    * Transformers (`transformers==4.53.0`).
    * Hugging Face Hub (`huggingface-hub==0.33.2`).
* **Vector Database:**
    * Weaviate (`weaviate-client==4.4.2`).
    * gRPC and Protobuf: (`grpcio==1.73.1`, `protobuf==6.31.1`, etc.).
* **User Interface Framework:**
    * Gradio (`gradio==5.35.0`).
* **Audio Processing Libraries:**
    * `ffmpy==0.6.0`.
    * `pydub==0.25.1`.
    * `audioread==3.0.1`, `librosa==0.11.0`, `soundfile==0.13.1`, `soxr==0.5.0.post1`.
* **Data Handling & Utilities:**
    * Pandas (`pandas==2.3.0`).
    * Redis (`redis==6.2.0`).
    * `nltk` (for sentence tokenization).
    * `accelerate==1.8.1`, `bitsandbytes==0.42.0`.
    * `pytest==8.4.1`.

## License

(Add your desired license information, e.g., MIT, Apache 2.0)

## Contributing

Contributions are welcome! Please follow standard GitHub practices: fork the repository, create a new branch, commit your changes, and open a pull request.
