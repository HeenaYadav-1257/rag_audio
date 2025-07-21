import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from sentence_transformers import SentenceTransformer
_ = SentenceTransformer("all-mpnet-base-v2")
import re
import gradio as gr
import redis
import warnings
import logging
import weaviate

from config import REDIS_HOST, REDIS_PORT, GRADIO_PORT, WEAVIATE_URL, AUDIO_DIR
from preprocess import load_audio, chunk_audio, get_audio_files
from features import extract_features
from transcribe import process_chunks
from embed import generate_embeddings
from vector_store import store_embeddings, query_embeddings
from llm import generate_response

warnings.filterwarnings("ignore", category=ResourceWarning)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# 🔁 Run once at launch
def index_all_audio():
    logger.info("📦 Preprocessing entire audio dataset...")
    audio_paths = get_audio_files()
    total = len(audio_paths)

    print(f"[🔍 Found {total} audio files in {AUDIO_DIR}]")

    success_count = 0
    failure_count = 0

    for i, audio_path in enumerate(audio_paths, 1):
        print(f"\n[{i}/{total}] 🚀 Processing: {os.path.basename(audio_path)}")

        try:
            audio = load_audio(audio_path)
            chunks = chunk_audio(audio)

            for chunk in chunks:
                extract_features(chunk)

            transcription = process_chunks(chunks)

            embeddings, sentences = generate_embeddings(transcription)

            file_ids = [os.path.basename(audio_path)] * len(sentences)
            chunk_ids = list(range(len(sentences)))

            store_embeddings(embeddings, sentences, file_ids, chunk_ids)

            print(f"✅ Done: {os.path.basename(audio_path)}")
            success_count += 1

        except Exception as e:
            print(f"❌ Failed: {os.path.basename(audio_path)} — {e}")
            failure_count += 1

    print("\n📊 Dataset Processing Complete")
    print(f"✅ {success_count} succeeded")
    print(f"❌ {failure_count} failed\n")


index_all_audio()  # <---- 🔥 RUN AT STARTUP


# 🧠 Answer Questions
def answer_query(query):
    try:
        logger.info("🔍 Answering query across all audio files...")
        cache_key = f"query:all:{query}"
        cached_response = r.get(cache_key)
        if cached_response:
            return cached_response.decode()

        connection_params = weaviate.connect.ConnectionParams.from_url(WEAVIATE_URL, grpc_port=50051)
        client = weaviate.WeaviateClient(connection_params=connection_params)
        client.connect()

        results = query_embeddings(query, client)
        context = " ".join([r["transcription"] for r in results["metadatas"] if r["transcription"]]).strip()

        if not context or len(context.split()) < 20:
            context = "Sorry, not enough relevant info was retrieved from the audios."

        response = generate_response(query, context)
        r.set(cache_key, response)

        images = fetch_relevant_images(response)
        return response, images


    except Exception as e:
        logger.error(f"[❌ ERROR]: {str(e)}")
        return f"[ERROR] {str(e)}"
    
def fetch_relevant_images(answer_text, image_folder="images"):
    keywords = re.findall(r"\b[A-Z][a-zA-Z0-9-]{2,}\b", answer_text)
    matched_images = []

    if not os.path.exists(image_folder):
        print(f"[⚠️] Image folder '{image_folder}' does not exist.")
        return []

    for keyword in keywords:
        for file in os.listdir(image_folder):
            if keyword.lower() in file.lower():
                matched_images.append(os.path.join(image_folder, file))
    
    return matched_images[:3] 

# 🎛️ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Ask Questions Across All Audio Files")

    with gr.Row():
        query_input = gr.Textbox(label="Ask your question", placeholder="e.g., What did Ocean Spirit report?")
        ask_btn = gr.Button("Get Answer")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer", lines=8)
        image_output = gr.Gallery(label="Matching Images", columns=3, object_fit="cover", height="auto")


    ask_btn.click(
        fn=answer_query,
        inputs=query_input,
        outputs=[answer_output, image_output]
    )

demo.launch(server_port=GRADIO_PORT)
