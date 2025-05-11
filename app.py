import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator

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

for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    answer = generate_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", answer))
    st.rerun()
