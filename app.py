import os
import tempfile
import streamlit as st
from gtts import gTTS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# -----------------------------------------------------------------------------
# Page Configuration & Professional Modern UI Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CivicPulse AI — Public Energy Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional UI Styling (Modern Teal/Slate Glassmorphism Palette)
st.markdown("""
    <style>
    /* Global Styling Imports & Resets */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Container */
    .hero-container {
        background: linear-gradient(135deg, #0F766E 0%, #042F2E 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 118, 110, 0.25);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #99F6E4;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
    }
    
    /* Custom Metric / Status Badges */
    .status-badge {
        display: inline-block;
        background-color: rgba(20, 184, 166, 0.15);
        border: 1px solid #14B8A6;
        color: #0D9488;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* Source Citation Cards */
    .citation-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #0D9488;
        padding: 14px 16px;
        border-radius: 10px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .citation-card:hover {
        border-left-color: #0F766E;
        background: #F1F5F9;
    }
    
    /* Modern Output Response Box */
    .response-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    
    /* Custom Sidebar Inputs & Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    .stButton>button {
        width: 100%;
        background-color: #0D9488;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0F766E;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top Hero Header Section
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="hero-container">
        <div class="status-badge">⚡ OSF × Andela Hackathon 2026 Submission</div>
        <div class="hero-title">CivicPulse AI</div>
        <div class="hero-subtitle">
            Demystifying public energy allocations, rural solar mini-grids, and transformer spending in plain, verifiable language with source citations and local audio playback.
        </div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Configuration & Control Panel
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste AI Studio Key")
    uploaded_file = st.file_uploader("Upload Budget/Procurement PDF", type=["pdf"])
    
    st.divider()
    st.markdown("### 🌐 Accessibility Settings")
    selected_language = st.selectbox(
        "Audio Dialect / Language",
        ["English", "Hausa", "Swahili", "Pidgin (Simplified English)", "French"]
    )
    
    st.divider()
    st.markdown("### 💡 Preset Energy Queries")
    st.caption("Click any sample query to test immediately:")
    
    sample_queries = [
        "What is the total allocation for rural solar mini-grids?",
        "Which wards are receiving new power transformers this year?",
        "What is the budget for solar streetlight maintenance?",
        "Are there any clean energy subsidies or grants listed?"
    ]
    
    selected_sample = None
    for q in sample_queries:
        if st.button(q, key=f"btn_{q}"):
            selected_sample = q

# -----------------------------------------------------------------------------
# Main Application Content Area
# -----------------------------------------------------------------------------
if uploaded_file and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    @st.cache_resource(show_spinner="Indexing document chunks for vector search...")
    def process_document(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore

    try:
        vectorstore = process_document(tmp_path)
        st.success("✅ Document processed and vector index created successfully!")
        
        # Determine query text
        default_text = selected_sample if selected_sample else ""
        user_query = st.text_input(
            "🔎 Search or ask a question about local energy spending:",
            value=default_text,
            placeholder="e.g., How much is allocated to solar streetlights in Ward B?"
        )
        
        if user_query:
            with st.spinner("Analyzing document and retrieving verified facts..."):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                
                # Context Retrieval
                retrieved_docs = retriever.invoke(user_query)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Infrastructure & Civic Prompt System
                prompt_template = f"""
                You are a professional civic intelligence assistant analyzing energy infrastructure spending in Africa.
                Target audience dialect/style: {selected_language}.

                Guidelines:
                1. Provide a clear, structured, plain-language breakdown. Avoid heavy economic jargon.
                2. Highlight monetary figures, project names, and target locations clearly in bullet points.
                3. Keep the overall response factual, accessible, and grounded solely in the provided context.

                Context from official document:
                {{context}}

                Question: {{question}}
                
                Answer (Plain Language Summary):
                """
                
                full_prompt = prompt_template.format(context=context_text, question=user_query)
                response = llm.invoke(full_prompt)
                
                # Two-Column Professional Dashboard View
                col1, col2 = st.columns([3, 2], gap="large")
                
                with col1:
                    st.markdown("#### 💡 Plain Language Analysis")
                    st.markdown(f'<div class="response-card">{response.content}</div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 🔊 Audio Summary (Low-Bandwidth Access)")
                    tts_lang = "sw" if selected_language == "Swahili" else ("fr" if selected_language == "French" else "en")
                    tts = gTTS(text=response.content[:350], lang=tts_lang, slow=False)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
                        tts.save(audio_file.name)
                        st.audio(audio_file.name, format="audio/mp3")

                with col2:
                    st.markdown("#### 📜 Ground Truth & Citation Proof")
                    st.caption("Verifiable quotes retrieved from the official PDF:")
                    for i, doc in enumerate(retrieved_docs):
                        page_num = doc.metadata.get("page", "N/A")
                        st.markdown(f"""
                        <div class="citation-card">
                            <strong style="color:#0F766E;">Source Chunk {i+1} — (Page {page_num}):</strong><br>
                            <span style="color:#475569; font-size:0.9rem;">"{doc.page_content[:260]}..."</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
    except Exception as e:
        st.error(f"Error processing document or query: {str(e)}")

elif not api_key:
    st.info("👈 Enter your Gemini API Key in the control panel on the left to activate.")
else:
    st.info("👈 Upload a budget or procurement PDF in the control panel to start analyzing allocations.")