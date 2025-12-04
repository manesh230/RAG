# app.py
import streamlit as st
import os
import json
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .medical-card {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .warning-card {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Medical RAG Assistant</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-card">
⚠️ <strong>Disclaimer:</strong> This AI assistant provides medical information for educational purposes only. 
It is not a substitute for professional medical advice, diagnosis, or treatment. 
Always consult with qualified healthcare providers for medical concerns.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965333.png", width=80)
    st.title("Configuration")
    
    # API Key input
    api_key = st.text_input("Gemini API Key", type="password", 
                           help="Enter your Google Gemini API key")
    
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This Medical RAG Assistant combines:
    - **MIMIC-IV** clinical data
    - **Knowledge graphs** for 25+ conditions
    - **Gemini AI** for reasoning
    - **RAG** for accurate medical information
    """)
    
    st.markdown("---")
    st.subheader("Conditions Covered")
    conditions = [
        "Acute Coronary Syndrome", "Alzheimer's", "Asthma", "Atrial Fibrillation",
        "COPD", "Diabetes", "Epilepsy", "Heart Failure", "Hypertension",
        "Migraine", "Parkinson's", "Pneumonia", "Sepsis", "Stroke",
        "Upper Gastrointestinal Bleeding"
    ]
    for condition in conditions:
        st.markdown(f"- {condition}")

# Classes
class SimpleDataProcessor:
    def __init__(self, base_path: str = "/content"):
        self.base_path = base_path
        self.kg_path = os.path.join(base_path, "mimic-iv-ext-direct-1.0", "mimic-iv-ext-direct-1.0.0", "diagnostic_kg", "Diagnosis_flowchart")
        self.cases_path = os.path.join(base_path, "mimic-iv-ext-direct-1.0", "mimic-iv-ext-direct-1.0.0", "Finished")

    def extract_knowledge(self):
        """Extract knowledge from KG files"""
        chunks = []
        
        try:
            for filename in os.listdir(self.kg_path):
                if not filename.endswith('.json'):
                    continue

                file_path = os.path.join(self.kg_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                condition = filename.replace('.json', '')
                knowledge = data.get('knowledge', {})

                for stage_name, stage_data in knowledge.items():
                    if isinstance(stage_data, dict):
                        if stage_data.get('Risk Factors'):
                            chunks.append({
                                'text': f"{condition} - Risk Factors: {stage_data['Risk Factors']}",
                                'metadata': {'type': 'knowledge', 'category': 'risk_factors', 'condition': condition}
                            })

                        if stage_data.get('Symptoms'):
                            chunks.append({
                                'text': f"{condition} - Symptoms: {stage_data['Symptoms']}",
                                'metadata': {'type': 'knowledge', 'category': 'symptoms', 'condition': condition}
                            })
        except Exception as e:
            st.error(f"Error extracting knowledge: {e}")
            
        return chunks

    def extract_patient_cases(self):
        """Extract patient cases and reasoning"""
        chunks = []

        try:
            for condition_folder in os.listdir(self.cases_path):
                condition_path = os.path.join(self.cases_path, condition_folder)
                if not os.path.isdir(condition_path):
                    continue

                for root, dirs, files in os.walk(condition_path):
                    for filename in files:
                        if not filename.endswith('.json'):
                            continue

                        file_path = os.path.join(root, filename)
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        case_id = filename.replace('.json', '')

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

                        for key in data:
                            if not key.startswith('input'):
                                reasoning = self._extract_reasoning(data[key])
                                if reasoning:
                                    chunks.append({
                                        'text': f"Case {case_id} - {condition_folder}\nReasoning:\n{reasoning}",
                                        'metadata': {'type': 'reasoning', 'case_id': case_id, 'condition': condition_folder}
                                    })
        except Exception as e:
            st.error(f"Error extracting cases: {e}")
            
        return chunks

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

class SimpleRAGSystem:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self._init_collections()

    def _init_collections(self):
        """Initialize collections"""
        self.knowledge_collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            embedding_function=self.embedding_function
        )
        self.cases_collection = self.client.get_or_create_collection(
            name="patient_cases",
            embedding_function=self.embedding_function
        )

    def index_data(self, chunks):
        """Index chunks into ChromaDB"""
        knowledge_docs, knowledge_metas, knowledge_ids = [], [], []
        case_docs, case_metas, case_ids = [], [], []

        for i, chunk in enumerate(chunks):
            if chunk['metadata']['type'] == 'knowledge':
                knowledge_docs.append(chunk['text'])
                knowledge_metas.append(chunk['metadata'])
                knowledge_ids.append(f"kg_{i}")
            else:
                case_docs.append(chunk['text'])
                case_metas.append(chunk['metadata'])
                case_ids.append(f"case_{i}")

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

        return len(knowledge_docs), len(case_docs)

    def query(self, question, top_k=5):
        """Query both collections"""
        knowledge_results = self.knowledge_collection.query(
            query_texts=[question],
            n_results=top_k
        )

        case_results = self.cases_collection.query(
            query_texts=[question],
            n_results=top_k
        )

        all_results = []
        if knowledge_results['documents']:
            all_results.extend(knowledge_results['documents'][0])
        if case_results['documents']:
            all_results.extend(case_results['documents'][0])

        return all_results

class MedicalAI:
    def __init__(self, rag_system, api_key):
        self.rag = rag_system
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def ask(self, question):
        """Generate answer using RAG context"""
        context_chunks = self.rag.query(question, top_k=5)
        context = "\n---\n".join(context_chunks)

        prompt = f"""You are a medical expert assistant. Use the following medical context to answer the question accurately and safely.

MEDICAL CONTEXT:
{context}

QUESTION: {question}

Provide a comprehensive, evidence-based medical answer based on the context. Include:
1. Key findings from the context
2. Relevant clinical information
3. Important considerations
4. Limitations of the information

If the context doesn't contain enough information, clearly state what information is missing.
Always remind the user to consult healthcare professionals for medical advice."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Main application
def main():
    # Initialize session state
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'medical_ai' not in st.session_state:
        st.session_state.medical_ai = None

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Ask Questions", "Data Status", "Sample Queries", "About"])

    with tab1:
        st.markdown('<h2 class="sub-header">🔍 Ask Medical Questions</h2>', unsafe_allow_html=True)
        
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar to use the AI assistant.")
        
        question = st.text_area(
            "Enter your medical question:",
            placeholder="e.g., What are the symptoms of migraine? How is chest pain evaluated?",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            ask_button = st.button("Ask AI", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("Clear", use_container_width=True)

        if clear_button:
            st.rerun()

        if ask_button:
            if not api_key:
                st.error("Please enter your Gemini API key first.")
            elif not question:
                st.error("Please enter a question.")
            else:
                with st.spinner("🧠 Processing your question..."):
                    try:
                        # Initialize system if needed
                        if not st.session_state.rag_initialized:
                            with st.status("Initializing system...", expanded=True) as status:
                                st.write("📂 Loading data...")
                                processor = SimpleDataProcessor()
                                kg_chunks = processor.extract_knowledge()
                                case_chunks = processor.extract_patient_cases()
                                st.session_state.chunks = kg_chunks + case_chunks
                                
                                st.write("🗄️ Setting up RAG database...")
                                rag_system = SimpleRAGSystem()
                                kg_count, case_count = rag_system.index_data(st.session_state.chunks)
                                
                                st.write("🤖 Connecting to Gemini AI...")
                                st.session_state.medical_ai = MedicalAI(rag_system, api_key)
                                st.session_state.rag_initialized = True
                                status.update(label="✅ System ready!", state="complete")

                        # Get answer
                        answer = st.session_state.medical_ai.ask(question)
                        
                        # Display answer
                        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
                        st.markdown("### 🤖 AI Response")
                        st.markdown(answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show context
                        with st.expander("📚 View Retrieved Context"):
                            rag_system = SimpleRAGSystem()
                            context_chunks = rag_system.query(question, top_k=3)
                            for i, chunk in enumerate(context_chunks):
                                st.markdown(f"**Context {i+1}:**")
                                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                                st.divider()
                                
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    with tab2:
        st.markdown('<h2 class="sub-header">📊 Data Status</h2>', unsafe_allow_html=True)
        
        if st.button("Check Data Status", key="check_data"):
            with st.spinner("Checking data..."):
                try:
                    processor = SimpleDataProcessor()
                    
                    # Count KG files
                    kg_files = len([f for f in os.listdir(processor.kg_path) if f.endswith('.json')])
                    
                    # Count case files
                    case_files = 0
                    for root, dirs, files in os.walk(processor.cases_path):
                        case_files += len([f for f in files if f.endswith('.json')])
                    
                    # Display stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Knowledge Graph Files", kg_files)
                    with col2:
                        st.metric("Patient Case Files", case_files)
                    
                    # Conditions list
                    st.subheader("Available Conditions")
                    conditions = [f.replace('.json', '') for f in os.listdir(processor.kg_path) if f.endswith('.json')]
                    for condition in sorted(conditions):
                        st.write(f"- {condition}")
                        
                except Exception as e:
                    st.error(f"Error checking data: {e}")

    with tab3:
        st.markdown('<h2 class="sub-header">💡 Sample Questions</h2>', unsafe_allow_html=True)
        
        sample_questions = [
            "What are the risk factors for acute coronary syndrome?",
            "How is atrial fibrillation diagnosed?",
            "What are the common symptoms of COPD?",
            "How is heart failure managed?",
            "What causes upper gastrointestinal bleeding?",
            "What are the diagnostic criteria for sepsis?",
            "How is diabetic ketoacidosis treated?",
            "What are the warning signs of a stroke?",
            "How is pneumonia differentiated from bronchitis?",
            "What are the treatment options for hypertension?"
        ]
        
        st.markdown("Try these sample questions:")
        for i, q in enumerate(sample_questions):
            if st.button(f"{i+1}. {q}", key=f"sample_{i}"):
                st.session_state.sample_question = q
                st.rerun()
        
        if 'sample_question' in st.session_state:
            st.text_input("Selected question:", value=st.session_state.sample_question, disabled=True)

    with tab4:
        st.markdown('<h2 class="sub-header">ℹ️ About This System</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🏥 Medical RAG Assistant
        
        This system combines Retrieval-Augmented Generation (RAG) with medical knowledge to provide 
        accurate, context-aware medical information.
        
        #### 📚 Data Sources
        - **MIMIC-IV Dataset**: Real clinical data from ICU patients
        - **Medical Knowledge Graphs**: Structured information for 25+ conditions
        - **Clinical Cases**: Patient narratives and diagnostic reasoning
        
        #### 🛠️ Technology Stack
        - **Streamlit**: Web interface
        - **ChromaDB**: Vector database for semantic search
        - **Sentence Transformers**: Text embeddings
        - **Google Gemini AI**: Large language model for reasoning
        
        #### 🔍 How It Works
        1. **Indexing**: Medical data is processed and stored in a vector database
        2. **Retrieval**: Relevant medical information is found based on your question
        3. **Generation**: AI synthesizes the retrieved information into a comprehensive answer
        4. **Presentation**: Answer is displayed with context and disclaimers
        
        #### ⚠️ Important Notes
        - This is an educational tool, not a medical device
        - Always verify information with healthcare professionals
        - Data is based on retrospective clinical records
        - AI responses should be critically evaluated
        """)

# Run the app
if __name__ == "__main__":
    main()
