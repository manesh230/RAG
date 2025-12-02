import streamlit as st
import pandas as pd
import json
from pathlib import Path
import re
from rank_bm25 import BM25Okapi
import tempfile
import zipfile
import requests
import os

# Set page configuration
st.set_page_config(
    page_title="Medical RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .disease-tag {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    .stButton > button {
        width: 100%;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MEDICAL RAG PIPELINE CLASS
# ============================================================================

class MedicalRAGPipeline:
    """Streamlit-compatible RAG pipeline for medical notes"""
    
    def __init__(self):
        self.notes_df = None
        self.documents_df = None
        self.retriever = None
    
    def load_disease_data(self, disease_dir: Path):
        """Load data for a specific disease"""
        records = []
        json_files = list(disease_dir.glob("*.json"))
        
        if not json_files:
            return None
        
        # Create progress indicator
        if 'progress' not in st.session_state:
            st.session_state.progress = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, fp in enumerate(json_files):
            # Update progress
            progress = (i + 1) / len(json_files)
            progress_bar.progress(progress)
            status_text.text(f"Loading {disease_dir.name}: {i+1}/{len(json_files)} files")
            
            try:
                with open(fp, "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                # Combine all inputs (input1 to input6)
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
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
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
        
        if len(df) == 0:
            return pd.DataFrame(chunks)
        
        # Create progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in df.iterrows():
            # Update progress
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Processing document {idx+1}/{len(df)}")
            
            text = self.clean_text(row['text'])
            if not text or len(text.split()) < 10:  # Skip very short texts
                continue
                
            words = text.split()
            start = 0
            chunk_num = 0
            
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = ' '.join(words[start:end])
                
                # Only keep chunks with reasonable length
                if len(chunk_text.split()) >= 10:
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
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(chunks)
    
    def build_retriever(self, documents_df: pd.DataFrame):
        """Build BM25 retriever"""
        with st.spinner("Building search index..."):
            # Prepare corpus for BM25
            corpus = []
            for text in documents_df['text'].tolist():
                if isinstance(text, str):
                    corpus.append(text.split())
                else:
                    corpus.append([])
            
            self.retriever = BM25Okapi(corpus)
            self.documents_df = documents_df.reset_index(drop=True)
        
        return self.retriever
    
    def search(self, query: str, k: int = 5):
        """Search for relevant documents"""
        if self.retriever is None:
            raise ValueError("Retriever not built. Call build_retriever first.")
        
        if not query or not query.strip():
            return []
        
        with st.spinner("Searching through documents..."):
            tokenized_query = query.split()
            
            # Handle empty query
            if not tokenized_query:
                return []
            
            scores = self.retriever.get_scores(tokenized_query)
            
            # Get top k indices
            top_indices = sorted(range(len(scores)), 
                               key=lambda i: scores[i], 
                               reverse=True)[:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents_df):
                    row = self.documents_df.iloc[idx]
                    results.append({
                        'doc_id': row['doc_id'],
                        'disease': row['disease'],
                        'file': row['file'],
                        'text': row['text'],
                        'score': float(scores[idx])
                    })
        
        return results

# ============================================================================
# DATASET DOWNLOADER
# ============================================================================

def download_and_extract_dataset(github_url: str):
    """Download and extract dataset from GitHub"""
    try:
        with st.spinner("Downloading dataset from GitHub..."):
            # Download the zip file
            response = requests.get(github_url)
            response.raise_for_status()
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "dataset.zip")
            
            # Save zip file
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract zip
            with st.spinner("Extracting dataset..."):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            
            # Find the main directory
            extracted_path = None
            for root, dirs, files in os.walk(temp_dir):
                if "Finished" in dirs:
                    extracted_path = Path(root)
                    break
            
            if extracted_path:
                return extracted_path
            else:
                # Try direct approach
                for item in Path(temp_dir).iterdir():
                    if item.is_dir() and "mimic" in item.name.lower():
                        return item
                
                return Path(temp_dir)
                
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download dataset: {str(e)}")
        return None
    except zipfile.BadZipFile:
        st.error("Downloaded file is not a valid zip file")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

# ============================================================================
# STREAMLIT APP
# ============================================================================

def initialize_rag_system():
    """Initialize the RAG system"""
    GITHUB_URL = "https://github.com/manesh230/RAG/raw/main/mimic-iv-ext-direct-1.0.0.zip"
    
    # Download and extract dataset
    data_dir = download_and_extract_dataset(GITHUB_URL)
    
    if not data_dir:
        st.error("Failed to download or extract dataset")
        return False
    
    # Check for Finished directory
    finished_dir = data_dir / "Finished"
    if not finished_dir.exists():
        # Try alternative paths
        for item in data_dir.iterdir():
            if item.is_dir() and "finished" in item.name.lower():
                finished_dir = item
                break
        
        if not finished_dir.exists():
            st.error(f"'Finished' directory not found in {data_dir}")
            st.write("Available directories:", [d.name for d in data_dir.iterdir() if d.is_dir()])
            return False
    
    # Get list of diseases
    disease_dirs = [d for d in finished_dir.iterdir() if d.is_dir()]
    
    if not disease_dirs:
        st.error("No disease directories found in 'Finished' folder")
        return False
    
    st.info(f"Found {len(disease_dirs)} disease directories")
    
    # Load all diseases
    all_dfs = []
    loaded_diseases = []
    
    for disease_dir in disease_dirs:
        with st.spinner(f"Loading {disease_dir.name} data..."):
            df = st.session_state.pipeline.load_disease_data(disease_dir)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                loaded_diseases.append(disease_dir.name)
    
    if not all_dfs:
        st.error("No data could be loaded from any disease directory")
        return False
    
    # Combine all data
    st.session_state.pipeline.notes_df = pd.concat(all_dfs, ignore_index=True)
    
    # Create document chunks
    with st.spinner("Creating document chunks for efficient searching..."):
        st.session_state.pipeline.documents_df = st.session_state.pipeline.chunk_documents(
            st.session_state.pipeline.notes_df,
            chunk_size=256,
            overlap=50
        )
    
    if len(st.session_state.pipeline.documents_df) == 0:
        st.error("No document chunks created. Check data quality.")
        return False
    
    # Build retriever
    st.session_state.pipeline.build_retriever(st.session_state.pipeline.documents_df)
    
    # Update session state
    st.session_state.initialized = True
    st.session_state.loaded_diseases = loaded_diseases
    st.session_state.data_stats = {
        'total_cases': len(st.session_state.pipeline.notes_df),
        'total_chunks': len(st.session_state.pipeline.documents_df),
        'disease_counts': st.session_state.pipeline.notes_df['disease'].value_counts().to_dict()
    }
    
    return True

def display_results(results):
    """Display search results in a nice format"""
    if not results:
        st.warning("No results found for your query.")
        return
    
    # Show summary stats
    diseases = [r['disease'] for r in results]
    unique_diseases = set(diseases)
    
    st.markdown(f"**Found {len(results)} relevant document(s) across {len(unique_diseases)} disease(s)**")
    
    # Display disease tags
    st.markdown("**Diseases in results:**")
    for disease in sorted(unique_diseases):
        st.markdown(f'<span class="disease-tag">{disease}</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Display each result
    for i, result in enumerate(results, 1):
        with st.container():
            st.markdown(f"### 📄 Result {i}: {result['disease']}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**File:** `{result['file']}`")
                st.markdown(f"**Relevance Score:** `{result['score']:.4f}`")
            with col2:
                if st.button(f"View Details {i}", key=f"view_{i}"):
                    st.session_state[f"show_details_{i}"] = not st.session_state.get(f"show_details_{i}", False)
            
            # Show text preview
            preview_text = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            st.markdown(f"**Preview:** {preview_text}")
            
            # Show full text if expanded
            if st.session_state.get(f"show_details_{i}", False):
                with st.expander("Full Document Text", expanded=True):
                    st.text(result['text'])
            
            st.divider()

def main():
    # App title
    st.markdown('<h1 class="main-header">🏥 Medical RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### Retrieve and Generate Answers from Clinical Medical Notes")
    
    # Initialize session state variables
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = MedicalRAGPipeline()
        st.session_state.initialized = False
        st.session_state.loaded_diseases = []
        st.session_state.data_stats = {}
        st.session_state.last_query = ""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        
        st.markdown("---")
        
        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 **Initialize RAG System**", type="primary", use_container_width=True):
                if initialize_rag_system():
                    st.rerun()
        else:
            st.markdown('<div class="success-box">✅ **System Initialized**</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📊 Dataset Statistics")
            
            if st.session_state.data_stats:
                st.metric("Total Cases", st.session_state.data_stats['total_cases'])
                st.metric("Document Chunks", st.session_state.data_stats['total_chunks'])
                
                st.markdown("**Cases by Disease:**")
                for disease, count in st.session_state.data_stats['disease_counts'].items():
                    st.write(f"- **{disease}:** {count} cases")
            
            st.markdown("---")
            
            # Quick query examples
            st.markdown("### 💡 Quick Queries")
            
            quick_queries = [
                "heart failure symptoms",
                "hypertension treatment",
                "COPD exacerbation",
                "chest pain diagnosis",
                "shortness of breath"
            ]
            
            for query in quick_queries:
                if st.button(f"🔍 {query}", key=f"quick_{query}"):
                    st.session_state.last_query = query
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This system retrieves information from 
        MIMIC-IV clinical notes using BM25 search.
        
        **Data Source:** GitHub repository with
        medical notes for various diseases.
        
        **Search Method:** BM25 algorithm for
        relevance-based document retrieval.
        """)
    
    # Main content area
    if not st.session_state.initialized:
        st.info("👈 **Please initialize the RAG system from the sidebar to begin.**")
        
        # Show sample queries
        st.markdown("### 📝 Sample Medical Queries")
        
        col1, col2, col3 = st.columns(3)
        
        sample_queries = [
            ("Cardiology", "chest pain with elevated troponin"),
            ("Pulmonology", "shortness of breath and wheezing"),
            ("Gastroenterology", "abdominal pain and diarrhea"),
            ("Nephrology", "hypertension and proteinuria"),
            ("Neurology", "headache with photophobia"),
            ("Endocrinology", "diabetes with polyuria")
        ]
        
        for i, (specialty, query) in enumerate(sample_queries):
            col = [col1, col2, col3][i % 3]
            with col:
                st.markdown(f"**{specialty}**")
                st.code(query)
        
        return
    
    # Search interface
    st.markdown('<div class="sub-header">🔍 Search Medical Notes</div>', unsafe_allow_html=True)
    
    # Query input with last query pre-filled
    query = st.text_area(
        "Enter your medical query:",
        value=st.session_state.get("last_query", ""),
        placeholder="e.g., What are the symptoms and treatments for heart failure?",
        height=100,
        key="query_input"
    )
    
    # Search parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        k_results = st.slider("Number of results", 1, 20, 10, help="How many documents to retrieve")
    with col2:
        min_score = st.slider("Minimum relevance score", 0.0, 50.0, 0.1, 0.1, 
                            help="Filter out low-scoring results")
    with col3:
        show_diseases = st.multiselect(
            "Filter by disease",
            options=st.session_state.loaded_diseases,
            default=st.session_state.loaded_diseases,
            help="Select diseases to include in search"
        )
    
    # Search button
    col1, col2 = st.columns([3, 1])
    with col2:
        search_button = st.button("🔎 **Search**", type="primary", use_container_width=True)
    
    # Perform search
    if search_button and query:
        st.session_state.last_query = query
        
        try:
            # Perform search
            all_results = st.session_state.pipeline.search(query, k=k_results * 2)  # Get more initially
            
            # Filter by disease if specified
            if show_diseases and len(show_diseases) < len(st.session_state.loaded_diseases):
                all_results = [r for r in all_results if r['disease'] in show_diseases]
            
            # Filter by minimum score
            filtered_results = [r for r in all_results if r['score'] >= min_score]
            
            # Limit to requested number
            final_results = filtered_results[:k_results]
            
            # Display results
            if final_results:
                display_results(final_results)
                
                # Export option
                st.markdown("---")
                if st.button("📥 Export Results to CSV"):
                    export_df = pd.DataFrame(final_results)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="medical_search_results.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No results found matching your criteria. Try broadening your search.")
                
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
    
    elif search_button and not query:
        st.warning("Please enter a query before searching.")

if __name__ == "__main__":
    main()
