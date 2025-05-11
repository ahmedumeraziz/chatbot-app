import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator

# Custom CSS for chat interface with taller input box
st.markdown("""
<style>
/* Remove form submit instructions */
.stFormSubmitContent {
    display: none;
}

/* Main chat messages area */
.chat-messages {
    padding-bottom: 100px;
}

/* Message bubbles */
.message {
    max-width: 80%;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 18px;
    line-height: 1.4;
    font-size: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.user-message {
    margin-left: auto;
    background-color: #007bff;
    color: white;
    border-bottom-right-radius: 4px;
}

.bot-message {
    margin-right: auto;
    background-color: #e9ecef;
    color: #212529;
    border-bottom-left-radius: 4px;
}

/* Fixed input footer */
.input-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    padding: 5px;
    border-top: 1px solid #e0e0e0;
    z-index: 1000;
}

/* Input wrapper */
.input-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 2px 15px 10px 15px;
    display: flex;
    gap: 8px;
    align-items: center;
}

/* Taller input field */
.message-input {
    flex: 1;
    padding: 10px 10px;
    border: 1px solid #ddd;
    border-radius: 24px;
    outline: none;
    font-size: 14px;
    line-height: 1.5;
    min-height: 68px;
    margin: 0;
    resize: vertical;
}

/* Send button */
.send-button {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background-color: #007bff;
    color: white;
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background-color 0.3s;
}

.send-button:hover {
    background-color: #0069d9;
}

/* Adjust Streamlit default styles */
.stApp {
    background-color: #f5f5f5 !important;
}
</style>
""", unsafe_allow_html=True)

# Load secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = "llama3-8b-8192"
GOOGLE_DOC_URL = "https://docs.google.com/document/d/196veS3lJcHJ7iJDSN47nnWO9XKHVoxBrSwtSCD8lvUM/edit?usp=sharing"

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "ready" not in st.session_state:
    st.session_state.ready = False

# Translation
def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang in ['ur', 'hi']:
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

# Google Docs text fetch
def get_text_from_google_doc(doc_url):
    doc_id = doc_url.split("/d/")[1].split("/")[0]
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(export_url)
    return response.text

# Chunking with overlap
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# TF-IDF retrieval
def get_relevant_chunks_tfidf(query, docs, k=3):
    vectorizer = TfidfVectorizer().fit(docs + [query])
    vectors = vectorizer.transform(docs + [query])
    sims = cosine_similarity(vectors[-1], vectors[:-1])[0]
    top_k = sims.argsort()[-k:][::-1]
    return [docs[i] for i in top_k]

# Smarter retrieval using exact match fallback
def get_relevant_chunks_smart(query, docs, k=3):
    keyword_matches = [doc for doc in docs if any(word.lower() in doc.lower() for word in query.split())]
    if keyword_matches:
        return keyword_matches[:k]
    return get_relevant_chunks_tfidf(query, docs, k)

# GROQ response generation
def generate_response(query):
    query = translate_to_english(query)
    context_chunks = get_relevant_chunks_smart(query, st.session_state.documents)
    context = "\n".join(context_chunks)

    prompt = f"""You are a highly intelligent CRM assistant. Use the following document context to answer the customer's question precisely. 

Context:
\"\"\"
{context}
\"\"\"

Question: {query}

Instructions:
- Provide a short, clear, professional response (1–2 sentences)
- Answer **only** from the above context
- Do **not** mention the document or repeat the question
- If answer is not availbale request customer to contact info@soopercart.com

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

# Prepare documents
if not st.session_state.ready:
    with st.spinner("Loading knowledge base..."):
        try:
            raw_text = get_text_from_google_doc(GOOGLE_DOC_URL)
            chunks = chunk_text(raw_text)
            st.session_state.documents = chunks
            st.session_state.ready = True
            st.success("Connected to Agent...")
        except Exception as e:
            st.error(f"Failed to load document: {e}")
            st.stop()

# UI
st.markdown(
    """<style>h4 {text-align: center;}</style>""",
    unsafe_allow_html=True,
)
st.markdown("#### Live Chat", unsafe_allow_html=True)

# Chat messages
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f'<div class="message user-message">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message bot-message">{msg}</div>', unsafe_allow_html=True)

# Fixed input footer with taller input box
with st.form("chat_form", clear_on_submit=True):
    st.markdown('<div class="input-footer">', unsafe_allow_html=True)
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    
    cols = st.columns([0.85, 0.15])
    with cols[0]:
        user_input = st.text_input(
            "Your message:",
            key="user_input",
            label_visibility="collapsed",
            placeholder="Type your message..."
        )
    with cols[1]:
        submitted = st.form_submit_button("Send", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Handle form submission
if submitted and user_input:
    answer = generate_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", answer))
    st.rerun()
