import streamlit as st
import pickle
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the model and vectorizer
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Main Container ── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 700px !important;
    }

    /* ── Hero Header Card ── */
    .hero-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem 2rem 2rem 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .hero-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        display: block;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-8px); }
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.55);
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
        letter-spacing: 0.3px;
    }

    /* ── Stats Row ── */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .stat-card {
        flex: 1;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        backdrop-filter: blur(6px);
    }

    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #a78bfa;
    }

    .stat-label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.45);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* ── Input Card ── */
    .input-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.8rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    .input-label {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* ── Streamlit textarea override ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1.5px solid rgba(167,139,250,0.3) !important;
        border-radius: 14px !important;
        color: #fff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 14px 16px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
        resize: vertical !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(167,139,250,0.8) !important;
        box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
        outline: none !important;
    }

    .stTextArea textarea::placeholder {
        color: rgba(255,255,255,0.25) !important;
    }

    .stTextArea label {
        color: rgba(255,255,255,0.75) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* ── Analyze Button ── */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
        margin-top: 0.5rem !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #4338ca) !important;
        box-shadow: 0 6px 28px rgba(124,58,237,0.6) !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Result Cards ── */
    .result-spam {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
        border: 1.5px solid rgba(239,68,68,0.5);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        animation: slideUp 0.4s ease;
        margin-top: 1rem;
    }

    .result-safe {
        background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.08));
        border: 1.5px solid rgba(52,211,153,0.5);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        animation: slideUp 0.4s ease;
        margin-top: 1rem;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .result-icon {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.6rem;
    }

    .result-title-spam {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f87171;
        margin: 0 0 0.4rem 0;
    }

    .result-title-safe {
        font-size: 1.6rem;
        font-weight: 800;
        color: #34d399;
        margin: 0 0 0.4rem 0;
    }

    .result-desc {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.5);
        margin: 0;
    }

    /* ── Warning override ── */
    .stAlert {
        background: rgba(251,191,36,0.1) !important;
        border: 1px solid rgba(251,191,36,0.3) !important;
        border-radius: 12px !important;
        color: #fbbf24 !important;
    }

    /* ── How it works section ── */
    .how-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.5rem;
    }

    .how-title {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }

    .step {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.6rem;
    }

    .step-num {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
    }

    .step-text {
        color: rgba(255,255,255,0.55);
        font-size: 0.83rem;
    }

    /* ── Footer ── */
    .footer-text {
        text-align: center;
        color: rgba(255,255,255,0.2);
        font-size: 0.75rem;
        margin-top: 1.5rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <span class="hero-icon">🛡️</span>
    <h1 class="hero-title">Spam Detector</h1>
    <p class="hero-subtitle">AI-powered SMS &amp; message spam detection using Machine Learning &amp; NLP</p>
</div>
""", unsafe_allow_html=True)

# ─── Stats Row ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-number">98%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">NLP</div>
        <div class="stat-label">Powered</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">TF-IDF</div>
        <div class="stat-label">Vectorizer</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">< 1s</div>
        <div class="stat-label">Detection</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Input Section ──────────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)

user_input = st.text_area(
    "✉️  Paste your message below",
    placeholder="e.g. Congratulations! You've won a free iPhone. Click here to claim now...",
    height=140,
    label_visibility="visible"
)

analyze_clicked = st.button("🔍  Analyze Message")

st.markdown('</div>', unsafe_allow_html=True)

# ─── Prediction Logic (unchanged) ───────────────────────────────────────────────
if analyze_clicked:
    if user_input.strip() == "":
        st.warning("⚠️  Please enter a message before analyzing.")
    else:
        with st.spinner("Analyzing your message..."):
            cleaned = clean_text(user_input)
            tfidf_vector = vectorizer.transform([cleaned])
            msg_length = np.array([[len(user_input)]])

            from scipy.sparse import hstack
            final_input = hstack([tfidf_vector, msg_length])

            prediction = model.predict(final_input)[0]

        if prediction == 1:
            st.markdown("""
            <div class="result-spam">
                <span class="result-icon">🚨</span>
                <h2 class="result-title-spam">Spam Detected!</h2>
                <p class="result-desc">This message shows strong indicators of spam. Be cautious and avoid clicking any links.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-safe">
                <span class="result-icon">✅</span>
                <h2 class="result-title-safe">Safe Message</h2>
                <p class="result-desc">This message appears to be legitimate. No spam patterns were detected.</p>
            </div>
            """, unsafe_allow_html=True)

# ─── How It Works ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-card">
    <div class="how-title">⚙️ How it works</div>
    <div class="step"><div class="step-num">1</div><div class="step-text">Your message is cleaned — lowercased, special chars removed, stop words filtered</div></div>
    <div class="step"><div class="step-num">2</div><div class="step-text">Text is lemmatized and vectorized using TF-IDF into numerical features</div></div>
    <div class="step"><div class="step-num">3</div><div class="step-text">ML model predicts Spam or Not Spam in under a second</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    Built with Streamlit · Scikit-learn · NLTK &nbsp;|&nbsp; Spam Detection Project
</div>
""", unsafe_allow_html=True)
