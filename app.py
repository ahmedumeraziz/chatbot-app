import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator

# ========== STREAMLIT CONFIGURATION ========== #
st.set_page_config(page_title="CRM Assistant", layout="wide")

# Hide Streamlit branding
hide_streamlit_style = """
<style>
footer {visibility: hidden;}
.stDeployButton {display:none;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ========== INITIALIZATION ========== #
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2").to("cpu")

embedder = load_embedder()

# Secrets and constants
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = "llama3-8b-8192"
GOOGLE_DOC_URL = "https://docs.google.com/document/d/196veS3lJcHJ7iJDSN47nnWO9XKHVoxBrSwtSCD8lvUM/edit?usp=sharing"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = []
if "ready" not in st.session_state:
    st.session_state.ready = False
if "error" not in st.session_state:
    st.session_state.error = None

# ========== HELPER FUNCTIONS ========== #
def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang in ['ur', 'hi']:
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

@st.cache_data
def get_text_from_google_doc(doc_url):
    try:
        doc_id = doc_url.split("/d/")[1].split("/")[0]
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        response = requests.get(export_url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.session_state.error = f"Failed to load document: {str(e)}"
        return None

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)

def get_relevant_chunks(query, k=3):
    query_embedding = embedder.encode([query])[0]
    scores = np.dot(st.session_state.doc_embeddings, query_embedding)
    top_k = np.argsort(scores)[-k:][::-1]
    return [st.session_state.documents[i] for i in top_k]

def generate_response(query):
    try:
        query = translate_to_english(query)
        context = "\n".join(get_relevant_chunks(query))
        
        prompt = f"""You are a helpful CRM assistant. Answer the customer's query based only on this context.

Context:
{context}

Instructions:
- Keep the answer short and clear (2–4 sentences max).
- Do NOT repeat the customer's question.
- Reply professionally and helpfully.

Answer:"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful CRM assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", 
            headers=headers, 
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ========== DOCUMENT LOADING ========== #
if not st.session_state.ready and not st.session_state.error:
    with st.status("Connecting to the Agent...", expanded=True) as status:
        try:
            st.write("Connecting to the Agent...")
            raw_text = get_text_from_google_doc(GOOGLE_DOC_URL)
            
            if raw_text is None:
                raise Exception(st.session_state.error)
                
            st.write("Connecting to the Agent...")
            chunks = chunk_text(raw_text)
            embeddings = embed_chunks(chunks)
            
            st.session_state.documents = chunks
            st.session_state.doc_embeddings = embeddings
            st.session_state.ready = True
            status.update(label="Connected to Agent!", state="complete", expanded=False)
        except Exception as e:
            st.session_state.error = str(e)
            status.update(label="Connection failed", state="error")

# ========== CHAT INTERFACE ========== #
st.title("CRM Assistant")

if st.session_state.error:
    st.error(f"Connection Error: {st.session_state.error}")
    if st.button("Retry Connection"):
        st.session_state.error = None
        st.session_state.ready = False
        st.rerun()

# Display chat history
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender.lower()):
        st.write(msg)

# User input
if prompt := st.chat_input("Your message"):
    if not st.session_state.ready:
        st.warning("Connecting to the Agent, please wait...")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append(("User", prompt))
        
        # Generate response
        with st.spinner("Agent is thinking..."):
            response = generate_response(prompt)
            st.session_state.chat_history.append(("AI", response))
        
        # Rerun to update chat
        st.rerun()
