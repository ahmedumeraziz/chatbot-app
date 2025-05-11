import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator

# Custom CSS for chat interface
st.markdown("""
<style>
/* Chat messages area */
.chat-messages {
    padding-bottom: 80px; /* Space for fixed input */
}

/* Message bubbles */
.message {
    max-width: 80%;
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 18px;
    line-height: 1.4;
    font-size: 14px;
}

.user-message {
    margin-left: auto;
    background-color: #007bff;
    color: white;
    border-bottom-right-radius: 4px;
}

.bot-message {
    margin-right: auto;
    background-color: #f0f0f0;
    color: #333;
    border-bottom-left-radius: 4px;
}

/* Fixed input footer */
.input-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    padding: 10px 0;
    border-top: 1px solid #e0e0e0;
    z-index: 1000;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
}

.input-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 0 15px;
    display: flex;
    gap: 8px;
    align-items: center;
}

.message-input {
    flex: 1;
    padding: 10px 12px;
    border: 1px solid #ddd;
    border-radius: 24px;
    outline: none;
    font-size: 14px;
    line-height: 1.5;
}

.message-input:focus {
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0,123,255,0.1);
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
    transition: background-color 0.2s;
}

.send-button:hover {
    background-color: #0069d9;
}

.send-icon {
    width: 20px;
    height: 20px;
}

/* Adjust Streamlit defaults */
.stApp {
    background-color: #ffffff !important;
}

.stTitle {
    padding: 15px 0 !important;
    margin-bottom: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ... [Keep all the previous Python code for functionality unchanged] ...

# UI
st.title("📞 CRM Assistant")

# Chat messages
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f'<div class="message user-message">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message bot-message">{msg}</div>', unsafe_allow_html=True)

# Fixed input footer
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
        submitted = st.form_submit_button("➤", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Handle form submission
if submitted and user_input:
    answer = generate_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", answer))
    st.rerun()
