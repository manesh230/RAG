# app.py
import streamlit as st
import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import requests
import zipfile
import tempfile
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Medical RAG Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E3A8A;
        padding: 20px;
    }
    .result-card {
        background-color: #f0f9ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #3b82f6;
    }
    .disease-tag {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.9em;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

class FastMedicalRAG:
    def __init__(self):
        self.documents_df = None
        self.retriever = None
        self.initialized = False
    
    def load_preprocessed_data(self, data_url=None):
        """Load preprocessed data from URL or local"""
        with st.spinner("Loading medical database..."):
            try:
                # For GitHub: Download preprocessed files
                if data_url:
                    # Download logic here
                    pass
                else:
                    # Load from local files (upload these to GitHub)
                    # You'll need to upload preprocessed_data.zip to GitHub
                    self.documents_df = pd.read_parquet("preprocessed_data/documents.parquet")
                    
                    with open("preprocessed_data/bm25_model.pkl", "rb") as f:
                        self.retriever = pickle.load(f)
                
                self.initialized = True
                return True
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return False
    
    def search(self, query: str, k: int = 5):
        """Fast search using pre-built index"""
        if not self.initialized or self.retriever is None:
            return []
        
        tokenized_query = query.split()
        if not tokenized_query:
            return []
        
        scores = self.retriever.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), 
                           key=lambda i: scores[i], 
                           reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents_df):
                row = self.documents_df.iloc[idx]
                results.append({
                    'disease': row['disease'],
                    'file': row['file'],
                    'text': row['text'],
                    'score': float(scores[idx])
                })
        
        return results

def main():
    st.markdown('<h1 class="main-title">🏥 Fast Medical RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### Quick Clinical Note Retrieval")
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = FastMedicalRAG()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if not st.session_state.rag.initialized:
            if st.button("🚀 Load Demo Data", type="primary"):
                # Load from local preprocessed files
                success = st.session_state.rag.load_preprocessed_data()
                if success:
                    st.success("✅ System loaded!")
                    st.rerun()
        
        if st.session_state.rag.initialized:
            st.success("✅ System Ready")
            st.metric("Documents", len(st.session_state.rag.documents_df))
            
            # Show disease distribution
            if st.session_state.rag.documents_df is not None:
                diseases = st.session_state.rag.documents_df['disease'].unique()
                st.write(f"**Diseases:** {', '.join(diseases)}")
            
            st.divider()
            
            # Quick queries
            st.header("💡 Quick Queries")
            quick_queries = [
                "high blood pressure",
                "chest pain",
                "shortness of breath",
                "cough fever",
                "heart failure"
            ]
            
            for q in quick_queries:
                if st.button(f"🔍 {q}"):
                    st.session_state.query = q
                    st.rerun()
    
    # Main content
    if not st.session_state.rag.initialized:
        st.info("👈 Click 'Load Demo Data' in the sidebar to start")
        
        # Show sample
        st.markdown("### This is a DEMO system")
        st.write("""
        For a production system with full data:
        1. Pre-process your data offline using `preprocess.py`
        2. Upload the preprocessed files to GitHub
        3. Modify the app to download from GitHub
        
        **Demo loads in < 5 seconds** with limited preprocessed data.
        """)
        return
    
    # Search interface
    query = st.text_input(
        "🔍 Search clinical notes:",
        value=getattr(st.session_state, 'query', ''),
        placeholder="e.g., symptoms of hypertension"
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        k_results = st.slider("Results to show", 1, 10, 5)
    
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                results = st.session_state.rag.search(query, k=k_results)
                
                if results:
                    st.subheader(f"📋 Found {len(results)} results")
                    
                    # Display results
                    for i, r in enumerate(results, 1):
                        with st.expander(f"Result {i}: {r['disease']} (Score: {r['score']:.4f})", expanded=(i==1)):
                            st.markdown(f"**File:** `{r['file']}`")
                            st.markdown(f"**Disease:** `{r['disease']}`")
                            st.markdown("**Text:**")
                            st.text(r['text'][:500] + "..." if len(r['text']) > 500 else r['text'])
                    
                    # Stats
                    st.subheader("📊 Statistics")
                    diseases = [r['disease'] for r in results]
                    disease_counts = pd.Series(diseases).value_counts()
                    
                    cols = st.columns(len(disease_counts))
                    for idx, (disease, count) in enumerate(disease_counts.items()):
                        with cols[idx]:
                            st.metric(disease, count)
                else:
                    st.warning("No results found. Try a different query.")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()
