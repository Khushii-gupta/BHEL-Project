import os
import pickle
import numpy as np
import faiss
from tkinter import messagebox
from sentence_transformers import SentenceTransformer

# Global variables
MATERIALS_FILE = 'materials'  # adjust as needed
INDEX_FILE = 'faiss.index'
EMBEDDING_DIM = 384               # adjust as needed

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index():
    try:
        if not os.path.exists(MATERIALS_FILE):
            messagebox.showerror("Error", f"Missing materials file: {MATERIALS_FILE}")
            return False

        with open(MATERIALS_FILE, 'rb') as f:
            materials = pickle.load(f)

        texts = [m['name'] for m in materials]  # adjust field name as needed
        embeddings = model.encode(texts, convert_to_numpy=True)

        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)

        messagebox.showinfo("Success", "FAISS index has been built successfully.")
        return True

    except Exception as e:
        messagebox.showerror("Error", f"Failed to build FAISS index:\n{e}")
        return False

if __name__ == "__main__":
    build_faiss_index()
