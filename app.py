import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator

# Load model and GROQ API key from secrets
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = embedder.to("cpu")  # Force CPU usage to avoid Streamlit Cloud GPU errors

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = "llama3-8b-8192"
GOOGLE_DOC_URL = "https://docs.google.com/document/d/196veS3lJcHJ7iJDSN47nnWO9XKHVoxBrSwtSCD8lvUM/edit?usp=sharing"

# Session setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = []
if "ready" not in st.session_state:
    st.session_state.ready = False

# Translation function
def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang in ['ur', 'hi']:
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

# Get Google Doc text
def get_text_from_google_doc(doc_url):
    doc_id = doc_url.split("/d/")[1].split("/")[0]
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(export_url)
    return response.text

# Chunk and embed
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

# Generate concise response
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

# Process the document
if not st.session_state.ready:
    with st.spinner("Connecting to the Agent..."):
        try:
            raw_text = get_text_from_google_doc(GOOGLE_DOC_URL)
            chunks = chunk_text(raw_text)
            embeddings = embed_chunks(chunks)
            st.session_state.documents = chunks
            st.session_state.doc_embeddings = embeddings
            st.session_state.ready = True
            st.success("✅ Connected to Sooper Cart AI")
        except Exception as e:
            st.error(f"❌ Failed to process document: {e}")

# --- Chat UI ---
st.markdown("""
    <style>
        .chat-wrapper {
            width: 300px;
            height: 500px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            border: 1px solid #ccc;
            border-radius: 10px;
            overflow: hidden;
            margin: auto;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .chat-container {
            flex: 1 1 auto;
            overflow-y: auto;
            padding: 10px;
            background-color: #f1f0f0;
        }

        .message {
            padding: 10px 14px;
            margin: 6px 0;
            border-radius: 18px;
            max-width: 90%;
            font-size: 14px;
            line-height: 1.4;
            word-wrap: break-word;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        .user {
            background-color: #0084ff;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 0;
        }

        .ai {
            background-color: #e5e5ea;
            color: black;
            align-self: flex-start;
            border-bottom-left-radius: 0;
        }

        .input-box {
            height: 60px;
            flex-shrink: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 8px;
            background: white;
            border-top: 1px solid #ccc;
        }

        .input-box input {
            width: 70%;
            padding: 8px 10px;
            font-size: 14px;
            border-radius: 20px;
            border: 1px solid #ccc;
        }

        .input-box button {
            padding: 8px 14px;
            margin-left: 8px;
            background-color: #0084ff;
            border: none;
            color: white;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

if st.session_state.ready:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for sender, msg in st.session_state.chat_history:
        role_class = "user" if sender == "You" else "ai"
        st.markdown(f'<div class="message {role_class}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <script>
            setTimeout(function() {
                var chatContainer = window.parent.document.querySelector('.chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }, 200);
        </script>
    """, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        st.markdown('<div class="input-box">', unsafe_allow_html=True)
        user_input = st.text_input("Type your message", placeholder="How may I help you?", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted and user_input:
        answer = generate_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", answer))
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
