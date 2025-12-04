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
GEMINI_API_KEY = "AIzaSyCkwbqccRPTUd3zLqJ3A6WagcdDRsMJQCY"

class DataExtractor:
    def __init__(self):
        self.zip_path = "./data.zip"
        self.extracted_path = "./data_extracted"
        self.github_url = "https://github.com/Mustehsan-Nisar-Rao/RAG/raw/main/mimic-iv-ext-direct-1.0.zip"
        
    def download_from_github(self):
        """Download ZIP file from GitHub"""
        try:
            st.info("📥 Downloading data from GitHub...")
            
            # Use raw GitHub URL
            response = requests.get(self.github_url, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with open(self.zip_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = int(50 * downloaded / total_size)
                                progress_bar.progress(min(progress, 100))
                                status_text.text(f"Downloaded {downloaded}/{total_size} bytes")
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Successfully downloaded data from GitHub")
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
            st.info("📦 Extracting ZIP file...")
            
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # Get file list and set up progress
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Extract all files
                for i, file in enumerate(file_list):
                    zip_ref.extract(file, self.extracted_path)
                    progress = int(100 * (i + 1) / total_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Extracting files... {i+1}/{total_files}")
                
                progress_bar.empty()
                status_text.empty()
            
            st.success("✅ Successfully extracted data from ZIP file")
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
        
        # Log found paths
        if self.kg_path:
            st.info(f"📁 Knowledge graph path: {self.kg_path}")
        if self.cases_path:
            st.info(f"📁 Cases path: {self.cases_path}")
    
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

        st.info(f"📊 Found {kg_count} knowledge files and {case_count} case files")
        return kg_count, case_count

    def extract_knowledge(self):
        """Extract knowledge from KG files"""
        chunks = []

        if not self.kg_path or not os.path.exists(self.kg_path):
            st.error(f"❌ Knowledge graph path not found")
            st.info(f"💡 Checked paths: {self.possible_kg_paths}")
            return chunks

        # Set up progress
        files = [f for f in os.listdir(self.kg_path) if f.endswith('.json')]
        total_files = len(files)
        
        if total_files == 0:
            st.warning("⚠️ No JSON files found in knowledge graph directory")
            return chunks
            
        progress_bar = st.progress(0)
        status_text = st.empty()

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
                
                # Update progress
                progress = int(100 * (i + 1) / total_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing knowledge files... {i+1}/{total_files}")
                
            except Exception as e:
                st.warning(f"⚠️ Error processing {filename}: {e}")
                continue

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Extracted {len(chunks)} knowledge chunks from {total_files} files")
        return chunks

    def extract_patient_cases(self):
        """Extract patient cases and reasoning"""
        chunks = []

        if not self.cases_path or not os.path.exists(self.cases_path):
            st.error(f"❌ Cases path not found")
            st.info(f"💡 Checked paths: {self.possible_case_paths}")
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
            st.warning("⚠️ No case files found")
            return chunks

        # Set up progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_files = 0
        for file_path, condition_folder in file_paths:
            self._process_case_file(file_path, condition_folder, chunks)
            processed_files += 1
            
            # Update progress
            progress = int(100 * processed_files / total_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing case files... {processed_files}/{total_files}")

        progress_bar.empty()
        status_text.empty()

        narratives = len([c for c in chunks if c['metadata']['type'] == 'narrative'])
        reasoning = len([c for c in chunks if c['metadata']['type'] == 'reasoning'])
        st.success(f"✅ Extracted {narratives} narrative chunks and {reasoning} reasoning chunks from {total_files} case files")
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
            st.warning(f"⚠️ Error processing {file_path}: {e}")

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
        st.info("🚀 Starting data extraction...")

        # Check if data exists
        kg_exists, cases_exists = self.check_data_exists()
        if not kg_exists and not cases_exists:
            st.error("❌ No valid data found after extraction.")
            st.info("💡 Please check the ZIP file structure")
            return []

        # Count files
        kg_count, case_count = self.count_files()

        if kg_count == 0 and case_count == 0:
            st.error("❌ No JSON files found in data directories.")
            return []

        # Extract data
        knowledge_chunks = self.extract_knowledge()
        case_chunks = self.extract_patient_cases()

        all_chunks = knowledge_chunks + case_chunks

        if all_chunks:
            st.success(f"🎯 Extraction complete: {len(knowledge_chunks)} knowledge + {len(case_chunks)} cases = {len(all_chunks)} total chunks")
        else:
            st.error("❌ No data chunks were extracted")

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

            st.success("✅ Created ChromaDB collections")
        except Exception as e:
            st.error(f"Error creating collections: {e}")

    def index_data(self):
        """Index all chunks into ChromaDB"""
        knowledge_docs, knowledge_metas, knowledge_ids = [], [], []
        case_docs, case_metas, case_ids = [], [], []

        try:
            total_chunks = len(self.chunks)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, chunk in enumerate(self.chunks):
                if chunk['metadata']['type'] == 'knowledge':
                    knowledge_docs.append(chunk['text'])
                    knowledge_metas.append(chunk['metadata'])
                    knowledge_ids.append(f"kg_{i}")
                else:
                    case_docs.append(chunk['text'])
                    case_metas.append(chunk['metadata'])
                    case_ids.append(f"case_{i}")

                # Update progress
                progress = int(100 * (i + 1) / total_chunks)
                progress_bar.progress(progress)
                status_text.text(f"Indexing chunks... {i+1}/{total_chunks}")

            progress_bar.empty()
            status_text.empty()

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

            st.success(f"✅ Indexed {len(knowledge_docs)} knowledge chunks and {len(case_docs)} case chunks")
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

            # Create prompt WITHOUT the "what's missing" section
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
    st.set_page_config(
        page_title="Medical RAG System",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Medical Diagnosis Assistant")
    st.markdown("Ask medical questions about symptoms, diagnoses, and patient cases")

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'medical_ai' not in st.session_state:
        st.session_state.medical_ai = None
    if 'data_extracted' not in st.session_state:
        st.session_state.data_extracted = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Show API key status (hardcoded, no input needed)
    st.sidebar.success("🔑 API key configured")
    
    # Data extraction section
    st.sidebar.subheader("📁 Data Setup")
    
    if not st.session_state.data_extracted:
        if st.sidebar.button("📥 Download & Extract Data", type="primary"):
            with st.spinner("Downloading data from GitHub and extracting..."):
                extractor = DataExtractor()
                if extractor.extract_data():
                    st.session_state.data_extracted = True
                    st.session_state.extractor = extractor
                    st.rerun()

    # Initialize system
    if st.session_state.data_extracted and not st.session_state.initialized:
        if st.sidebar.button("🚀 Initialize System", type="primary"):
            try:
                with st.spinner("🚀 Processing medical data and setting up RAG system... This may take a few minutes."):
                    # Initialize processor and extract data
                    processor = SimpleDataProcessor(st.session_state.extractor.extracted_path)
                    chunks = processor.run()

                    if not chunks:
                        st.error("❌ No data was extracted. Please check your data file structure.")
                        return

                    # Initialize RAG system
                    rag_system = SimpleRAGSystem(chunks)
                    rag_system.create_collections()
                    rag_system.index_data()

                    # Initialize Medical AI with hardcoded API key
                    st.session_state.medical_ai = MedicalAI(rag_system, GEMINI_API_KEY)
                    st.session_state.rag_system = rag_system
                    st.session_state.initialized = True

                st.success("✅ System initialized successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")

    # Main interface
    if st.session_state.initialized and st.session_state.medical_ai:
        st.header("💬 Medical Query Interface")

        # Question input
        question = st.text_area(
            "Enter your medical question:",
            placeholder="e.g., What are the symptoms of migraine? How is chest pain evaluated? What are risk factors for gastrointestinal bleeding?",
            height=100
        )

        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Number of context chunks", min_value=1, max_value=10, value=5)
            with col2:
                show_context = st.checkbox("Show retrieved context", value=False)

        if st.button("Get Medical Answer", type="primary", use_container_width=True) and question:
            with st.spinner("🔍 Analyzing medical context and generating answer..."):
                try:
                    # Get answer
                    answer = st.session_state.medical_ai.ask(question)

                    # Display answer
                    st.subheader("🤖 Medical Answer")
                    st.markdown(f"**Question:** {question}")
                    st.markdown("**Answer:**")
                    st.write(answer)

                    # Show context if requested
                    if show_context:
                        st.subheader("📚 Retrieved Context")
                        context_chunks = st.session_state.rag_system.query(question, top_k=top_k)
                        
                        for i, chunk in enumerate(context_chunks):
                            with st.expander(f"Context Chunk {i+1}"):
                                st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

        # Example questions
        st.subheader("💡 Example Questions")
        examples = [
            "What are the diagnostic criteria for migraine?",
            "How is chest pain evaluated in emergency settings?",
            "What are common risk factors for gastrointestinal bleeding?",
            "Describe the symptoms and diagnosis process for pneumonia",
            "What are the treatment options for asthma?",
            "How to diagnose and manage diabetes?"
        ]

        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, use_container_width=True):
                    st.session_state.last_question = example
                    st.rerun()

        # System info
        with st.expander("📊 System Information"):
            if st.session_state.rag_system:
                knowledge_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'knowledge'])
                narrative_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'narrative'])
                reasoning_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'reasoning'])
                
                st.write(f"**Knowledge chunks:** {knowledge_count}")
                st.write(f"**Case narratives:** {narrative_count}")
                st.write(f"**Case reasoning:** {reasoning_count}")
                st.write(f"**Total chunks:** {len(st.session_state.rag_system.chunks)}")

    else:
        st.info("""
        👋 **Welcome to the Medical RAG System!**
        
        To get started:
        1. 📥 Click 'Download & Extract Data' to get medical data from GitHub
        2. 🚀 Click 'Initialize System' to build the RAG system
        
        *API key is pre-configured*
        *Data source: https://github.com/Mustehsan-Nisar-Rao/RAG/raw/main/mimic-iv-ext-direct-1.0.zip*
        """)

if __name__ == "__main__":
    main()
