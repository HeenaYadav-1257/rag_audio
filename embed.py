from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import os
import pickle

# ✅ Load sentence tokenizer directly
punkt_path = "/Users/heenayadav/nltk_data/tokenizers/punkt/english.pickle"
with open(punkt_path, "rb") as f:
    tokenizer = pickle.load(f)

def sent_tokenize(text):
    return tokenizer.tokenize(text)

# ✅ Load local model
MODEL_DIR = os.path.join(os.getcwd(), "all-mpnet-base-v2")
model = SentenceTransformer(MODEL_DIR, device="mps" if torch.backends.mps.is_available() else "cpu")

def generate_embeddings(full_transcript):
    if isinstance(full_transcript, list):
        full_transcript = " ".join(full_transcript)

    sentences = sent_tokenize(full_transcript)
    sentences = [s.strip().replace("\n", " ") for s in sentences if len(s.strip()) > 5]

    embeddings = model.encode(sentences)

    print("[🧠 Total Sentences]:", len(sentences))
    if sentences:
        print("[🧠 First Sentence]:", sentences[0])
        print("[🧠 First Vector Preview]:", embeddings[0][:5])
    else:
        print("[⚠️] No valid sentences for embedding!")

    return embeddings, sentences
