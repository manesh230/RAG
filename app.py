import streamlit as st
import os
import json
import tempfile
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import requests
import zipfile
import io

# Your hardcoded API key
GEMINI_API_KEY = "AIzaSyBWOR9PZZuHm4IADT4qw9xIPT4NFQru780"

class DataExtractor:
    def __init__(self):
        self.zip_path = "./data.zip"
        self.extracted_path = "./data_extracted"
        self.github_url = "https://github.com/manesh230/RAG/blob/main/mimic-iv-ext-direct-1.0.0.zip?raw=true"
        
    def download_from_github(self):
        """Download ZIP file from GitHub"""
        try:
            # Use raw GitHub URL with ?raw=true parameter
            response = requests.get(self.github_url, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                with open(self.zip_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                return True
            else:
                st.error(f"❌ Failed to download file. HTTP Status: {response.status_code}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error downloading from GitHub: {e}")
            return False
        
    def extract_data(self):
        """Extract data from ZIP file"""
        # First, download the file if it doesn't exist
        if not os.path.exists(self.zip_path):
            if not self.download_from_github():
                return False
            
        try:
            # Create extraction directory
            os.makedirs(self.extracted_path, exist_ok=True)
            
            # Extract ZIP file
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # Get file list
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                
                # Extract all files
                for i, file in enumerate(file_list):
                    zip_ref.extract(file, self.extracted_path)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error extracting ZIP file: {e}")
            return False

class SimpleDataProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        # Try different possible paths after extraction
        self.possible_kg_paths = [
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "mimic-iv-ext-direct-1.0.0", "diagnostic_kg", "Diagnosis_flowchart"),
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "diagnostic_kg", "Diagnosis_flowchart"),
            os.path.join(base_path, "diagnostic_kg", "Diagnosis_flowchart"),
            os.path.join(base_path, "Diagnosis_flowchart"),
            os.path.join(base_path, "mimic-iv-ext-direct-1.0.0", "diagnostic_kg", "Diagnosis_flowchart"),
        ]
        self.possible_case_paths = [
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "mimic-iv-ext-direct-1.0.0", "Finished"),
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "Finished"),
            os.path.join(base_path, "Finished"),
            os.path.join(base_path, "cases"),
            os.path.join(base_path, "mimic-iv-ext-direct-1.0.0", "Finished"),
        ]
        
        self.kg_path = self._find_valid_path(self.possible_kg_paths)
        self.cases_path = self._find_valid_path(self.possible_case_paths)
    
    def _find_valid_path(self, possible_paths):
        """Find the first valid path that exists"""
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def check_data_exists(self):
        """Check if data directories exist and have files"""
        kg_exists = self.kg_path and os.path.exists(self.kg_path) and any(f.endswith('.json') for f in os.listdir(self.kg_path))
        cases_exists = self.cases_path and os.path.exists(self.cases_path) and any(os.path.isdir(os.path.join(self.cases_path, d)) for d in os.listdir(self.cases_path))
        
        return kg_exists, cases_exists

    def count_files(self):
        """Count all JSON files"""
        kg_count = 0
        if self.kg_path and os.path.exists(self.kg_path):
            kg_count = len([f for f in os.listdir(self.kg_path) if f.endswith('.json')])

        case_count = 0
        if self.cases_path and os.path.exists(self.cases_path):
            for item in os.listdir(self.cases_path):
                item_path = os.path.join(self.cases_path, item)
                if os.path.isdir(item_path):
                    for root, dirs, files in os.walk(item_path):
                        case_count += len([f for f in files if f.endswith('.json')])
                elif item.endswith('.json'):
                    case_count += 1

        return kg_count, case_count

    def extract_knowledge(self):
        """Extract knowledge from KG files"""
        chunks = []

        if not self.kg_path or not os.path.exists(self.kg_path):
            return chunks

        files = [f for f in os.listdir(self.kg_path) if f.endswith('.json')]
        total_files = len(files)
        
        if total_files == 0:
            return chunks

        for i, filename in enumerate(files):
            file_path = os.path.join(self.kg_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                condition = filename.replace('.json', '')
                knowledge = data.get('knowledge', {})

                for stage_name, stage_data in knowledge.items():
                    if isinstance(stage_data, dict):
                        # Extract risk factors
                        if stage_data.get('Risk Factors'):
                            chunks.append({
                                'text': f"{condition} - Risk Factors: {stage_data['Risk Factors']}",
                                'metadata': {'type': 'knowledge', 'category': 'risk_factors', 'condition': condition}
                            })

                        # Extract symptoms
                        if stage_data.get('Symptoms'):
                            chunks.append({
                                'text': f"{condition} - Symptoms: {stage_data['Symptoms']}",
                                'metadata': {'type': 'knowledge', 'category': 'symptoms', 'condition': condition}
                            })
                
            except Exception as e:
                continue

        return chunks

    def extract_patient_cases(self):
        """Extract patient cases and reasoning"""
        chunks = []

        if not self.cases_path or not os.path.exists(self.cases_path):
            return chunks

        # Count total files for progress
        total_files = 0
        file_paths = []
        
        for item in os.listdir(self.cases_path):
            item_path = os.path.join(self.cases_path, item)
            if os.path.isdir(item_path):
                for root, dirs, files in os.walk(item_path):
                    json_files = [f for f in files if f.endswith('.json')]
                    total_files += len(json_files)
                    for f in json_files:
                        file_paths.append((os.path.join(root, f), item))
            elif item.endswith('.json'):
                total_files += 1
                file_paths.append((item_path, "General"))

        if total_files == 0:
            return chunks

        for file_path, condition_folder in file_paths:
            self._process_case_file(file_path, condition_folder, chunks)

        return chunks

    def _process_case_file(self, file_path, condition_folder, chunks):
        """Process individual case file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filename = os.path.basename(file_path)
            case_id = filename.replace('.json', '')

            # Extract narrative (inputs)
            narrative_parts = []
            for i in range(1, 7):
                key = f'input{i}'
                if key in data and data[key]:
                    narrative_parts.append(f"{key}: {data[key]}")

            if narrative_parts:
                chunks.append({
                    'text': f"Case {case_id} - {condition_folder}\nNarrative:\n" + "\n".join(narrative_parts),
                    'metadata': {'type': 'narrative', 'case_id': case_id, 'condition': condition_folder}
                })

            # Extract reasoning
            for key in data:
                if not key.startswith('input'):
                    reasoning = self._extract_reasoning(data[key])
                    if reasoning:
                        chunks.append({
                            'text': f"Case {case_id} - {condition_folder}\nReasoning:\n{reasoning}",
                            'metadata': {'type': 'reasoning', 'case_id': case_id, 'condition': condition_folder}
                        })
        except Exception as e:
            pass

    def _extract_reasoning(self, data):
        """Simple reasoning extraction"""
        reasoning_lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                if '$Cause_' in key:
                    reasoning_text = key.split('$Cause_')[0].strip()
                    if reasoning_text:
                        reasoning_lines.append(reasoning_text)

                if isinstance(value, (dict, list)):
                    nested_reasoning = self._extract_reasoning(value)
                    if nested_reasoning:
                        reasoning_lines.append(nested_reasoning)

        elif isinstance(data, list):
            for item in data:
                nested_reasoning = self._extract_reasoning(item)
                if nested_reasoning:
                    reasoning_lines.append(nested_reasoning)

        return "\n".join(reasoning_lines) if reasoning_lines else ""

    def run(self):
        """Run complete extraction"""
        # Check if data exists
        kg_exists, cases_exists = self.check_data_exists()
        if not kg_exists and not cases_exists:
            return []

        # Extract data
        knowledge_chunks = self.extract_knowledge()
        case_chunks = self.extract_patient_cases()

        all_chunks = knowledge_chunks + case_chunks

        return all_chunks

class SimpleRAGSystem:
    def __init__(self, chunks, db_path="./chroma_db"):
        self.chunks = chunks
        self.db_path = db_path
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.client = chromadb.PersistentClient(path=db_path)
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")

    def create_collections(self):
        """Create separate collections for knowledge and cases"""
        try:
            # Knowledge collection
            self.knowledge_collection = self.client.get_or_create_collection(
                name="medical_knowledge",
                embedding_function=self.embedding_function
            )

            # Cases collection
            self.cases_collection = self.client.get_or_create_collection(
                name="patient_cases",
                embedding_function=self.embedding_function
            )

        except Exception as e:
            st.error(f"Error creating collections: {e}")

    def index_data(self):
        """Index all chunks into ChromaDB"""
        knowledge_docs, knowledge_metas, knowledge_ids = [], [], []
        case_docs, case_metas, case_ids = [], [], []

        try:
            for i, chunk in enumerate(self.chunks):
                if chunk['metadata']['type'] == 'knowledge':
                    knowledge_docs.append(chunk['text'])
                    knowledge_metas.append(chunk['metadata'])
                    knowledge_ids.append(f"kg_{i}")
                else:
                    case_docs.append(chunk['text'])
                    case_metas.append(chunk['metadata'])
                    case_ids.append(f"case_{i}")

            # Add to collections
            if knowledge_docs:
                self.knowledge_collection.add(
                    documents=knowledge_docs,
                    metadatas=knowledge_metas,
                    ids=knowledge_ids
                )

            if case_docs:
                self.cases_collection.add(
                    documents=case_docs,
                    metadatas=case_metas,
                    ids=case_ids
                )

        except Exception as e:
            st.error(f"Error indexing data: {e}")

    def query(self, question, top_k=5):
        """Simple query across both collections"""
        try:
            # Query knowledge
            knowledge_results = self.knowledge_collection.query(
                query_texts=[question],
                n_results=top_k
            )

            # Query cases
            case_results = self.cases_collection.query(
                query_texts=[question],
                n_results=top_k
            )

            # Combine results
            all_results = []
            if knowledge_results['documents']:
                all_results.extend(knowledge_results['documents'][0])
            if case_results['documents']:
                all_results.extend(case_results['documents'][0])

            return all_results
        except Exception as e:
            st.error(f"Error querying RAG system: {e}")
            return []

class MedicalAI:
    def __init__(self, rag_system, api_key):
        self.rag = rag_system
        try:
            genai.configure(api_key=api_key)
            # Use a more widely available model
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")

    def ask(self, question):
        try:
            # Get relevant context from RAG
            context_chunks = self.rag.query(question, top_k=5)
            context = "\n---\n".join(context_chunks)

            # Create prompt
            prompt = f"""You are a medical expert. Use the following medical context to answer the question accurately and comprehensively.

MEDICAL CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive medical answer based on the context. Focus on the information available in the context."""

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"

def main():
    # Custom CSS for modern medical-themed GUI
    st.set_page_config(
        page_title="MediSearch AI | Medical Diagnosis Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    st.markdown("""
    <style>
    /* Main styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border-left: 5px solid #4a90e2;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a2980, #26d0ce);
        color: white;
    }
    
    .sidebar-header {
        padding: 1.5rem 1rem;
        text-align: center;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-header h2 {
        color: white;
        font-size: 1.5rem;
        margin: 0;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-active {
        background: #4CAF50;
        color: white;
    }
    
    .status-inactive {
        background: #f44336;
        color: white;
    }
    
    .status-warning {
        background: #FF9800;
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1a2980, #26d0ce);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
    }
    
    .stTextArea textarea:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a2980;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Chat bubble styling */
    .user-bubble {
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        color: white;
        padding: 1rem;
        border-radius: 20px 20px 3px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
    }
    
    .ai-bubble {
        background: white;
        color: #333;
        padding: 1rem;
        border-radius: 20px 20px 20px 3px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        border: 1px solid #e0e0e0;
    }
    
    /* Icon styling */
    .icon-large {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    /* Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1>🩺 MediSearch AI</h1>
            <p>Advanced Medical Diagnosis Assistant powered by RAG Technology</p>
            <div style="margin-top: 1rem;">
                <span class="status-badge status-active">AI-Powered</span>
                <span class="status-badge status-warning">Clinical Knowledge Base</span>
                <span class="status-badge status-active">Real-time Analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'medical_ai' not in st.session_state:
        st.session_state.medical_ai = None
    if 'data_extracted' not in st.session_state:
        st.session_state.data_extracted = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar - Modern Design
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>⚙️ System Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 📊 System Status")
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if st.session_state.data_extracted:
                st.markdown('<div class="metric-card"><div class="metric-value">✓</div><div class="metric-label">Data Ready</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><div class="metric-value">⏳</div><div class="metric-label">Data Pending</div></div>', unsafe_allow_html=True)
        
        with status_col2:
            if st.session_state.initialized:
                st.markdown('<div class="metric-card"><div class="metric-value">✓</div><div class="metric-label">AI Ready</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><div class="metric-value">⏳</div><div class="metric-label">AI Pending</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data Management Section
        st.markdown("### 📁 Data Management")
        
        if not st.session_state.data_extracted:
            if st.button("📥 Download & Process Dataset", use_container_width=True):
                with st.spinner("🚀 Downloading medical dataset..."):
                    extractor = DataExtractor()
                    if extractor.extract_data():
                        st.session_state.data_extracted = True
                        st.session_state.extractor = extractor
                        st.success("✅ Dataset downloaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to download dataset")
        else:
            st.success("✅ Dataset ready!")
        
        # System Initialization
        st.markdown("### 🚀 System Initialization")
        
        if st.session_state.data_extracted and not st.session_state.initialized:
            if st.button("⚡ Initialize RAG System", use_container_width=True):
                with st.spinner("🧠 Building medical knowledge base..."):
                    try:
                        # Initialize processor and extract data
                        processor = SimpleDataProcessor(st.session_state.extractor.extracted_path)
                        chunks = processor.run()

                        if not chunks:
                            st.error("❌ No data was extracted.")
                            return

                        # Initialize RAG system
                        rag_system = SimpleRAGSystem(chunks)
                        rag_system.create_collections()
                        rag_system.index_data()

                        # Initialize Medical AI
                        st.session_state.medical_ai = MedicalAI(rag_system, GEMINI_API_KEY)
                        st.session_state.rag_system = rag_system
                        st.session_state.initialized = True

                        st.success("✅ System initialized successfully!")
                        st.balloons()
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown("---")
        
        # Quick Stats
        if st.session_state.initialized and st.session_state.rag_system:
            st.markdown("### 📈 Knowledge Base Stats")
            knowledge_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'knowledge'])
            narrative_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'narrative'])
            reasoning_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'reasoning'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Knowledge", f"{knowledge_count}")
                st.metric("Cases", f"{narrative_count}")
            with col2:
                st.metric("Reasoning", f"{reasoning_count}")
                st.metric("Total", f"{len(st.session_state.rag_system.chunks)}")
        
        st.markdown("---")
        
        # API Status
        st.markdown("### 🔑 API Status")
        st.success("✅ Gemini API Active")
        st.info("Model: Gemini 2.5 Flash")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.8rem;">
            MediSearch AI v2.0<br>
            Clinical Decision Support System
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    if st.session_state.initialized and st.session_state.medical_ai:
        # Chat Interface
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #1a2980; margin-bottom: 1rem;">💬 Clinical Query Interface</h3>
            <p style="color: #666; margin-bottom: 1.5rem;">Ask medical questions about symptoms, diagnoses, treatments, and patient management</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat history display
        if st.session_state.chat_history:
            st.markdown("### 📜 Conversation History")
            for chat in st.session_state.chat_history[-5:]:  # Show last 5 messages
                if chat['role'] == 'user':
                    st.markdown(f'<div class="user-bubble">{chat["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ai-bubble">{chat["content"]}</div>', unsafe_allow_html=True)
            st.markdown("---")
        
        # Question input with enhanced styling
        question = st.text_area(
            "**Enter your medical query:**",
            placeholder="Example: What are the diagnostic criteria for myocardial infarction? Or: How to manage a patient with acute asthma exacerbation?",
            height=120,
            key="query_input"
        )
        
        # Control Panel
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            analyze_button = st.button("🔍 Analyze & Generate Report", use_container_width=True, type="primary")
        with col2:
            top_k = st.slider("Context Depth", 1, 10, 5, help="Number of medical documents to reference")
        with col3:
            show_context = st.checkbox("Show Sources", value=False)
        
        if analyze_button and question:
            # Add to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': question
            })
            
            # Generate response
            with st.spinner("🤖 Analyzing medical context..."):
                try:
                    # Get answer
                    answer = st.session_state.medical_ai.ask(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': answer
                    })
                    
                    # Display in chat format
                    st.markdown(f'<div class="user-bubble">{question}</div>', unsafe_allow_html=True)
                    
                    with st.expander("📋 View Detailed Medical Analysis", expanded=True):
                        st.markdown(f"""
                        <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #1a2980;">
                        {answer}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show context if requested
                    if show_context:
                        with st.expander("🔬 View Retrieved Medical Sources"):
                            context_chunks = st.session_state.rag_system.query(question, top_k=top_k)
                            for i, chunk in enumerate(context_chunks):
                                with st.expander(f"Medical Source {i+1}", expanded=False):
                                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating analysis: {str(e)}")
        
        # Quick Query Buttons
        st.markdown("---")
        st.markdown("### 🚀 Quick Medical Queries")
        
        quick_queries = [
            "Diagnostic criteria for sepsis",
            "Management of type 2 diabetes",
            "Stroke risk factors and prevention",
            "Pneumonia treatment guidelines",
            "Hypertension management protocol",
            "GERD symptoms and treatment"
        ]
        
        cols = st.columns(3)
        for i, query in enumerate(quick_queries):
            with cols[i % 3]:
                if st.button(f"📌 {query}", use_container_width=True):
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': query
                    })
                    
                    with st.spinner("Analyzing..."):
                        answer = st.session_state.medical_ai.ask(query)
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': answer
                        })
                    st.rerun()
        
        # Features Section
        st.markdown("---")
        st.markdown("### ✨ System Features")
        
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        with feat_col1:
            st.markdown("""
            <div class="custom-card">
                <div class="icon-large">📚</div>
                <h4>Comprehensive Knowledge</h4>
                <p>Access extensive medical literature and case studies</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            st.markdown("""
            <div class="custom-card">
                <div class="icon-large">⚡</div>
                <h4>Real-time Analysis</h4>
                <p>Instant retrieval and analysis of medical information</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col3:
            st.markdown("""
            <div class="custom-card">
                <div class="icon-large">🛡️</div>
                <h4>Clinical Accuracy</h4>
                <p>Evidence-based responses from verified medical sources</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Conversation History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        # Welcome/Setup Screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="padding: 3rem; background: white; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                <h2 style="color: #1a2980; margin-bottom: 1.5rem;">Welcome to MediSearch AI 🩺</h2>
                <p style="font-size: 1.2rem; line-height: 1.6; color: #444; margin-bottom: 2rem;">
                    Your intelligent medical diagnosis assistant powered by advanced RAG technology. 
                    Get instant access to medical knowledge, diagnostic guidelines, and case-based learning.
                </p>
                
                <div style="background: linear-gradient(90deg, #1a2980, #26d0ce); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
                    <h4 style="margin: 0;">🚀 Quick Setup Guide</h4>
                </div>
                
                <div style="display: flex; align-items: center; margin-bottom: 1.5rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <div style="background: #1a2980; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">1</div>
                    <div>
                        <h5 style="margin: 0; color: #1a2980;">Download Medical Dataset</h5>
                        <p style="margin: 0.25rem 0 0 0; color: #666;">Click "Download & Process Dataset" in the sidebar</p>
                    </div>
                </div>
                
                <div style="display: flex; align-items: center; margin-bottom: 1.5rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <div style="background: #26d0ce; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">2</div>
                    <div>
                        <h5 style="margin: 0; color: #1a2980;">Initialize AI System</h5>
                        <p style="margin: 0.25rem 0 0 0; color: #666;">Click "Initialize RAG System" to build knowledge base</p>
                    </div>
                </div>
                
                <div style="display: flex; align-items: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <div style="background: #4CAF50; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">3</div>
                    <div>
                        <h5 style="margin: 0; color: #1a2980;">Start Asking Questions</h5>
                        <p style="margin: 0.25rem 0 0 0; color: #666;">Enter medical queries in the chat interface</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 2rem; background: white; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); height: 100%;">
                <h4 style="color: #1a2980; text-align: center; margin-bottom: 2rem;">📊 System Readiness</h4>
                
                <div style="text-align: center; margin-bottom: 2rem;">
                    <div style="width: 150px; height: 150px; margin: 0 auto; position: relative;">
                        <div style="width: 150px; height: 150px; border-radius: 50%; background: conic-gradient(#4CAF50 0% 30%, #f0f0f0 30% 100%); display: flex; align-items: center; justify-content: center;">
                            <div style="width: 120px; height: 120px; border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: bold; color: #1a2980;">
                                30%
                            </div>
                        </div>
                    </div>
                    <p style="margin-top: 1rem; color: #666;">Setup Progress</p>
                </div>
                
                <div style="margin-bottom: 1.5rem;">
                    <p style="font-weight: bold; color: #1a2980; margin-bottom: 0.5rem;">🔗 Data Source</p>
                    <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 10px; font-size: 0.9rem; color: #666;">
                        MIMIC-IV Extended Dataset
                    </div>
                </div>
                
                <div style="margin-bottom: 1.5rem;">
                    <p style="font-weight: bold; color: #1a2980; margin-bottom: 0.5rem;">🧠 AI Model</p>
                    <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 10px; font-size: 0.9rem; color: #666;">
                        Gemini 2.5 Flash
                    </div>
                </div>
                
                <div>
                    <p style="font-weight: bold; color: #1a2980; margin-bottom: 0.5rem;">📁 Database</p>
                    <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 10px; font-size: 0.9rem; color: #666;">
                        ChromaDB Vector Store
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Features showcase
        st.markdown("---")
        st.markdown("## ✨ Key Features")
        
        feature_cols = st.columns(4)
        features = [
            ("🔍", "Intelligent Search", "Advanced semantic search across medical literature"),
            ("⚡", "Fast Response", "Real-time answers to clinical questions"),
            ("📚", "Comprehensive", "Access to thousands of medical cases"),
            ("🎯", "Accurate", "Evidence-based medical information")
        ]
        
        for idx, (icon, title, desc) in enumerate(features):
            with feature_cols[idx]:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); height: 100%;">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">{icon}</div>
                    <h4 style="color: #1a2980; margin-bottom: 0.5rem;">{title}</h4>
                    <p style="color: #666; font-size: 0.9rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Call to action
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(90deg, #1a2980, #26d0ce); border-radius: 20px; margin-top: 2rem;">
            <h2 style="color: white; margin-bottom: 1rem;">Ready to Transform Medical Research?</h2>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-bottom: 2rem;">
                Get started with just two clicks! Setup your medical knowledge base now.
            </p>
            <div style="font-size: 3rem;" class="pulse">👇</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
