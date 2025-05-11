import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator

# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2").to("cpu")

# Secrets & constants
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = "llama3-8b-8192"
GOOGLE_DOC_URL = "https://docs.google.com/document/d/196veS3lJcHJ7iJDSN47nnWO9XKHVoxBrSwtSCD8lvUM/edit?usp=sharing"

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = []
if "ready" not in st.session_state:
    st.session_state.ready = False

# Text translation
def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang in ['ur', 'hi']:
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

# Document handling
def get_text_from_google_doc(doc_url):
    doc_id = doc_url.split("/d/")[1].split("/")[0]
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(export_url)
    return response.text

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

# Response generation
def generate_response(query):
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

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Load document on first run
if not st.session_state.ready:
    with st.spinner("Loading CRM knowledge base..."):
        try:
            raw_text = get_text_from_google_doc(GOOGLE_DOC_URL)
            chunks = chunk_text(raw_text)
            embeddings = embed_chunks(chunks)
            st.session_state.documents = chunks
            st.session_state.doc_embeddings = embeddings
            st.session_state.ready = True
        except Exception as e:
            st.error(f"❌ Failed to process document: {e}")

# Styling to match modern chat
st.markdown("""
    <style>
    .chat-box {
        max-width: 420px;
        height: 600px;
        margin: 0 auto;
        background: #f5f5f5;
        border: 1px solid #ccc;
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .chat-history {
        flex: 1;
        padding: 15px;
        overflow-y: auto;
        background-color: #fff;
    }
    .message {
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 16px;
        max-width: 80%;
        font-size: 14px;
        line-height: 1.4;
        word-wrap: break-word;
        display: inline-block;
    }
    .user-msg {
        background-color: #0084ff;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    .ai-msg {
        background-color: #e1e1e1;
        color: black;
        margin-right: auto;
        border-bottom-left-radius: 0;
    }
    .input-area {
        padding: 10px;
        background: #f1f1f1;
        display: flex;
        border-top: 1px solid #ddd;
    }
    .input-area input[type="text"] {
        flex: 1;
        padding: 10px 16px;
        border-radius: 24px 0 0 24px;
        border: 1px solid #ccc;
        font-size: 14px;
        outline: none;
    }
    .input-area button {
        background: #0084ff;
        color: white;
        border: none;
        padding: 0 18px;
        font-size: 18px;
        border-radius: 0 24px 24px 0;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# --- CHAT UI ---
if st.session_state.ready:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)

    # Chat history display
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for sender, msg in st.session_state.chat_history:
        css_class = "user-msg" if sender == "You" else "ai-msg"
        st.markdown(f'<div class="message {css_class}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        st.markdown('<div class="input-area">', unsafe_allow_html=True)
        user_input = st.text_input("Message", placeholder="Type message here...", label_visibility="collapsed")
        submitted = st.form_submit_button("➤")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted and user_input:
        answer = generate_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", answer))
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
