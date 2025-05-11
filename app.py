import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator

# Load the embedder (no .to("cpu"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load secrets safely
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit secrets.")
    st.stop()

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

# Translate non-English input
def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang in ['ur', 'hi']:
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except Exception:
        return text

# Get text from Google Doc
def get_text_from_google_doc(doc_url):
    try:
        doc_id = doc_url.split("/d/")[1].split("/")[0]
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        response = requests.get(export_url)
        if response.status_code == 200:
            return response.text
        else:
            raise ValueError("Unable to access Google Doc. Make sure it is public.")
    except Exception as e:
        raise ValueError(f"Failed to fetch document: {e}")

# Chunking and embedding
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)

def get_relevant_chunks(query, k=3):
    if len(st.session_state.doc_embeddings) == 0:
        return []
    query_embedding = embedder.encode([query])[0]
    scores = np.dot(st.session_state.doc_embeddings, query_embedding)
    top_k = np.argsort(scores)[-k:][::-1]
    return [st.session_state.documents[i] for i in top_k]

# Generate answer from GROQ API
def generate_response(query):
    query = translate_to_english(query)
    relevant_chunks = get_relevant_chunks(query)
    if not relevant_chunks:
        return "Sorry, I couldn't find any relevant information to answer your question."

    context = "\n".join(relevant_chunks)
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
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Load document and prepare embeddings
if not st.session_state.ready:
    with st.spinner("Loading knowledge base..."):
        try:
            raw_text = get_text_from_google_doc(GOOGLE_DOC_URL)
            chunks = chunk_text(raw_text)
            embeddings = embed_chunks(chunks)
            st.session_state.documents = chunks
            st.session_state.doc_embeddings = embeddings
            st.session_state.ready = True
            st.success("Connected to Agent.")
        except Exception as e:
            st.error(f"Failed to load document: {e}")
            st.stop()

# Chat interface
#st.title("📞 CRM Assistant")

# Display chat history
for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:")
    submitted = st.form_submit_button("Send")

# Handle response
if submitted and user_input:
    answer = generate_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", answer))
    st.rerun()
