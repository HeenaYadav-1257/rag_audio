from llama_cpp import Llama
from config import MODEL_PATH  # Define this in config.py
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

# Initialize llama.cpp model
llm = Llama(
    model_path=MODEL_PATH,     # Path to .gguf model
    n_ctx=4096,                # Context size
    n_threads=4,               # Adjust based on CPU cores
    use_mlock=False,
    verbose=False
)

def generate_response(query, context):
    prompt = f"""
    You are an intelligent assistant designed to answer questions based on transcribed audio conversations.

    Use the following transcript (from one or more audios) to provide a clear, specific, and helpful answer.

    Always stay grounded in the context. If the information is missing or unclear, say so confidently.

    ---
    Transcript:
    {context}

    Question:
    {query}

    Answer:
    """


    output = llm(prompt, max_tokens=100, stop=["###", "</s>"])
    return output["choices"][0]["text"].strip()
    print("[DEBUG] LLM output:", output)

