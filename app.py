import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator

# Custom CSS for chat interface
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    height: 70vh;
    width: 100%;
    margin: 0 auto;
    background-color: #fafafa;
    border-radius: 10px;
    position: relative;
    border: 1px solid #e0e0e0;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 18px;
    line-height: 1.4;
    font-size: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.user-message {
    align-self: flex-end;
    background-color: #007bff;
    color: white;
    border-bottom-right-radius: 4px;
}

.bot-message {
    align-self: flex-start;
    background-color: #e9ecef;
    color: #212529;
    border-bottom-left-radius: 4px;
}

.input-area {
    position: sticky;
    bottom: 0;
    padding: 15px;
    background-color: white;
    border-top: 1px solid #ddd;
    display: flex;
    gap: 10px;
    align-items: center;
}

.message-input {
    flex: 1;
    padding: 12px 15px;
    border: 1px solid #ddd;
    border-radius: 24px;
    outline: none;
    font-size: 14px;
}

.message-input:focus {
    border-color: #007bff;
}

.send-button {
    width: 40px;
    height: 40px;
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

.send-icon {
    width: 20px;
    height: 20px;
}

.stSpinner > div {
    text-align: center;
    margin-top: 20px;
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
- Provide a short, clear, professional response (2–4 sentences).
- Answer **only** from the above context.
- Do **not** mention the document or repeat the question.

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
st.title("📞 CRM Assistant")

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f'<div class="message user-message">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message bot-message">{msg}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-messages

# Input area
st.markdown("""
<div class="input-area">
    <form id="chat_form" class="stForm" style="width: 100%; display: flex; gap: 10px; align-items: center;">
        <input type="text" name="user_input" class="message-input" placeholder="Type your message..." autocomplete="off">
        <button type="submit" class="send-button">
            <svg class="send-icon" viewBox="0 0 24 24">
                <path fill="currentColor" d="M2,21L23,12L2,3V10L17,12L2,14V21Z" />
            </svg>
        </button>
    </form>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container

# Handle form submission
if st._is_running_with_streamlit:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
    if ctx and hasattr(ctx, 'form_data'):
        form_data = ctx.form_data
        if 'chat_form' in form_data:
            user_input = form_data['chat_form']['user_input']
            if user_input:
                answer = generate_response(user_input)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("AI", answer))
                st.rerun()
