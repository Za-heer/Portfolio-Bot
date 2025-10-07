import os
import json
import math
import pickle
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

import numpy as np
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    from sklearn.metrics.pairwise import cosine_similarity

HF_TOKEN = os.getenv("HF_API_TOKEN")
EMBED_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = os.getenv("RAG_INDEX_DIR", "data/faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
META_FILE = os.path.join(INDEX_DIR, "meta.pkl")
EMB_FILE = os.path.join(INDEX_DIR, "embeddings.npy")

os.makedirs(INDEX_DIR, exist_ok=True)

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def load_portfolio(path="data/portfolio.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    for i in range(0, max(1, len(tokens)), step):
        chunk = tokens[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
    return chunks

def docs_from_portfolio(portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Produce a list of docs with explicit, consistent formatting.
    Special-handles 'projects' to produce strong, descriptive text per project.
    """
    docs = []
    idx = 0
    for section, value in portfolio.items():
        sec = str(section)
        # Special handling for projects list of dicts
        if sec.lower() == "projects" and isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    name = item.get("name", "").strip()
                    desc = item.get("description", "").strip()
                    techs = item.get("technologies", []) or item.get("tech", []) or []
                    techs_str = ", ".join(techs) if isinstance(techs, (list, tuple)) else str(techs)
                    text = f"PROJECT: {name}. DESCRIPTION: {desc}. TECHNOLOGIES: {techs_str}"
                else:
                    text = str(item)
                # Do not over-chunk project single-line; keep each project as 1-2 chunks
                chunks = chunk_text(text, chunk_size=220, overlap=40)
                docs += [{"id": (idx := idx+1), "section": sec, "text": t.strip()} for t in chunks]
            continue

        # Generic handling
        if isinstance(value, str):
            text = value
            docs += [{"id": (idx := idx+1), "section": sec, "text": t.strip()} for t in chunk_text(text)]
        elif isinstance(value, dict):
            # join fields into a single descriptive text per dict
            combined = " | ".join(f"{k}: {v}" for k, v in value.items())
            docs += [{"id": (idx := idx+1), "section": sec, "text": t.strip()} for t in chunk_text(combined)]
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    line = " | ".join(f"{kk}: {vv}" for kk, vv in item.items())
                else:
                    line = str(item)
                docs += [{"id": (idx := idx+1), "section": sec, "text": t.strip()} for t in chunk_text(line)]
        else:
            docs += [{"id": (idx := idx+1), "section": sec, "text": str(value)}]
    return docs

def call_hf_embeddings(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    Call HF router feature-extraction pipeline and return list of embeddings.
    Uses: https://router.huggingface.co/hf-inference/models/{EMBED_MODEL}/pipeline/feature-extraction
    """
    if not HF_TOKEN:
        raise RuntimeError("Set HF_API_TOKEN in environment to call Hugging Face Inference API.")
    url = f"https://router.huggingface.co/hf-inference/models/{EMBED_MODEL}/pipeline/feature-extraction"
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = requests.post(url, headers=HEADERS, json={"inputs": batch}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # common response shapes:
        # - list of lists (one embedding per input)
        # - list of dicts with "embedding" keys
        if isinstance(data, list) and len(data) and isinstance(data[0], list):
            embeddings += data
        elif isinstance(data, list) and isinstance(data[0], dict) and "embedding" in data[0]:
            embeddings += [d["embedding"] for d in data]
        else:
            raise RuntimeError(f"Unexpected embedding response: {data}")
    return embeddings

class RAGStore:
    def __init__(self, portfolio_path="data/portfolio.json"):
        self.portfolio_path = portfolio_path
        self.index = None
        self.meta = []  # list of dicts: {"id": int, "section": str, "text": str}
        self.dim = None
        self.embeddings = None  # numpy array (N, dim) of raw embeddings

    def build_index(self, rebuild: bool = False):
        portfolio = load_portfolio(self.portfolio_path)
        docs = docs_from_portfolio(portfolio)
        texts = [d["text"] for d in docs]
        print(f"[RAG] Chunked into {len(texts)} pieces (documents). Example samples:")
        for i, sample in enumerate(texts[:6], 1):
            print(f"  [{i}] {sample[:200]}{'...' if len(sample)>200 else ''}")
        embeddings = call_hf_embeddings(texts)
        arr = np.array(embeddings).astype("float32")
        # Save raw embeddings for debugging/searching fallback
        np.save(EMB_FILE, arr)
        self.embeddings = arr
        if _HAS_FAISS:
            # For FAISS inner-product search, normalize vectors to use cosine-sim via inner product
            arr_norm = arr.copy()
            faiss.normalize_L2(arr_norm)
            self.dim = arr_norm.shape[1]
            index = faiss.IndexFlatIP(self.dim)
            index.add(arr_norm)
            faiss.write_index(index, INDEX_FILE)
            with open(META_FILE, "wb") as f:
                pickle.dump(docs, f)
            self.index = index
            self.meta = docs
            print("[RAG] FAISS index built and saved.")
        else:
            # fallback: store embeddings and meta for linear search
            with open(META_FILE, "wb") as f:
                pickle.dump({"embeddings": arr, "meta": docs}, f)
            self.meta = docs
            print("[RAG] FAISS not available. Saved embeddings for fallback search.")
        print(f"[RAG] Saved {len(docs)} meta entries and embeddings shape {arr.shape}")

    def load_index(self):
        # load FAISS index if present, else fallback meta/embeddings
        if _HAS_FAISS and os.path.exists(INDEX_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                self.meta = pickle.load(f)
            self.dim = self.index.d
            # attempt to load saved embeddings.npy if available
            if os.path.exists(EMB_FILE):
                try:
                    self.embeddings = np.load(EMB_FILE)
                    print(f"[RAG] Loaded embeddings from {EMB_FILE} shape={self.embeddings.shape}")
                except Exception:
                    self.embeddings = None
            print("[RAG] FAISS index loaded.")
        elif os.path.exists(META_FILE):
            with open(META_FILE, "rb") as f:
                data = pickle.load(f)
            # fallback format: dict with embeddings + meta OR plain meta list
            if isinstance(data, dict) and "embeddings" in data:
                self.embeddings = data["embeddings"]
                self.meta = data["meta"]
                print("[RAG] Fallback embeddings loaded from meta file.")
            else:
                self.meta = data
                # try to load embeddings.npy
                if os.path.exists(EMB_FILE):
                    self.embeddings = np.load(EMB_FILE)
                    print(f"[RAG] Loaded embeddings from {EMB_FILE} shape={self.embeddings.shape}")
                print("[RAG] Meta loaded (no faiss index).")
        else:
            raise FileNotFoundError("No index or meta found. Run build_index first.")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        emb = call_hf_embeddings([query_text])[0]
        q = np.array([emb]).astype("float32")
        results = []
        if _HAS_FAISS and self.index is not None:
            faiss.normalize_L2(q)
            D, I = self.index.search(q, top_k)
            for score, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                meta = self.meta[idx]
                results.append({"score": float(score), "section": meta.get("section"), "text": meta.get("text")})
        else:
            arr = getattr(self, "embeddings", None)
            if arr is None:
                raise RuntimeError("No embeddings available for search (build index first).")
            # compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(q, arr)[0]
            top_idx = sims.argsort()[::-1][:top_k]
            for idx in top_idx:
                results.append({"score": float(sims[idx]), "section": self.meta[idx].get("section"), "text": self.meta[idx].get("text")})
        return results

    def debug_query(self, query_text: str, top_k: int = 10):
        """
        Debug helper: show top_k matches + scores for a given query.
        Prints results to stdout and returns them as a list.
        """
        print(f"[RAG debug] Query: {query_text!r}")
        emb = call_hf_embeddings([query_text])[0]
        q = np.array([emb]).astype("float32")

        out = []
        if _HAS_FAISS and self.index is not None:
            faiss.normalize_L2(q)
            D, I = self.index.search(q, top_k)
            for score, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                meta = self.meta[idx]
                print(f"  score={float(score):.4f}  section={meta.get('section')}  text={meta.get('text')[:220]}...")
                out.append({"score": float(score), "section": meta.get("section"), "text": meta.get("text")})
        else:
            arr = getattr(self, "embeddings", None)
            if arr is None:
                raise RuntimeError("No embeddings loaded for debug. Run build_index first.")
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(q, arr)[0]
            top_idx = sims.argsort()[::-1][:top_k]
            for idx in top_idx:
                score = float(sims[idx])
                meta = self.meta[idx]
                print(f"  score={score:.4f}  section={meta.get('section')}  text={meta.get('text')[:220]}...")
                out.append({"score": score, "section": meta.get("section"), "text": meta.get("text")})
        return out

if __name__ == "__main__":
    store = RAGStore()
    # build or load index depending on files
    try:
        store.load_index()
    except Exception as e:
        print("Index not found or load failed:", e)
        print("Building index now...")
        store.build_index()

    print("\n[RAG CLI] Ready. Try a debug query.")
    while True:
        q = input("\nEnter query (or 'exit'): ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        store.debug_query(q, top_k=8)
