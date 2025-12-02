import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm

class MedicalRAGPipeline:
    """Modular RAG pipeline for medical notes"""
    
    def __init__(self):
        self.notes_df = None
        self.documents_df = None
        self.retriever = None
    
    def load_disease_data(self, disease_dir: Path):
        """Load data for a specific disease"""
        records = []
        json_files = list(disease_dir.glob("*.json"))
        
        for fp in tqdm(json_files, desc=f"Loading {disease_dir.name}"):
            with open(fp, "r") as f:
                data = json.load(f)
            
            # Combine all inputs
            inputs = []
            for i in range(1, 7):
                key = f"input{i}"
                val = data.get(key, "")
                if isinstance(val, str) and val.strip():
                    inputs.append(val.strip())
            
            full_text = "\n\n".join(inputs).strip()
            if full_text:
                records.append({
                    "file": fp.name,
                    "disease": disease_dir.name,
                    "text": full_text,
                })
        
        return pd.DataFrame(records) if records else None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def chunk_documents(self, df: pd.DataFrame, chunk_size: int = 256, overlap: int = 50):
        """Chunk documents into smaller pieces"""
        chunks = []
        
        for idx, row in df.iterrows():
            text = self.clean_text(row['text'])
            if not text:
                continue
                
            words = text.split()
            start = 0
            chunk_num = 0
            
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = ' '.join(words[start:end])
                
                chunks.append({
                    'doc_id': f"{idx}_{chunk_num}",
                    'disease': row['disease'],
                    'file': row['file'],
                    'text': chunk_text,
                    'original_index': idx
                })
                
                if end == len(words):
                    break
                    
                start = end - overlap
                chunk_num += 1
        
        return pd.DataFrame(chunks)
    
    def build_retriever(self, documents_df: pd.DataFrame):
        """Build BM25 retriever"""
        corpus = [doc.split() for doc in documents_df['text'].tolist()]
        self.retriever = BM25Okapi(corpus)
        self.documents_df = documents_df.reset_index(drop=True)
        return self.retriever
    
    def search(self, query: str, k: int = 5):
        """Search for relevant documents"""
        if self.retriever is None:
            raise ValueError("Retriever not built. Call build_retriever first.")
        
        tokenized_query = query.split()
        scores = self.retriever.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(scores)), 
                           key=lambda i: scores[i], 
                           reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            row = self.documents_df.iloc[idx]
            results.append({
                'doc_id': row['doc_id'],
                'disease': row['disease'],
                'file': row['file'],
                'text': row['text'],
                'score': float(scores[idx])
            })
        
        return results
