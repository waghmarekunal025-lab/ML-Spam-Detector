import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set elegant browser tab parameters
st.set_page_config(page_title="ML Spam Detector", page_icon="🛡️", layout="centered")

# --- BACKGROUND COMPUTE ENGINE (Cached to run instantly) ---
@st.cache_resource
def initialize_and_train_model():
    # Load dataset corpus
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
    )
    
    # Vectorization Matrix Extraction
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Fit Classifier Core Engine
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

# Run background setup handshake
model, vectorizer = initialize_and_train_model()

# --- PROFESSIONAL GRAPHICAL USER INTERACTION LAYER ---
st.title("🛡️ Predictive AI: SMS & Email Spam Detector")
st.caption("A supervised Machine Learning classification engine powered by an optimized Scikit-Learn pipeline.")

st.markdown("---")

# User Text Target Box Area
user_input = st.text_area(
    "📥 Enter or paste your message text string below:",
    placeholder="Type something here... e.g., Winner of a free cash reward! or Are we meeting up today?",
    height=120
)

if st.button("Analyze Security Vector", type="primary"):
    if user_input.strip() == "":
        st.warning("Please type or paste a valid message sequence first!")
    else:
        # Transform unstructured string array format to fit training dimensions
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]
        confidence_percentage = probability[prediction] * 100
        
        st.markdown("### 🔮 Classification Verdict:")
        
        # Display crisp visual block alerts based on predictive outcomes
        if prediction == 1:
            st.error(f"### 🚨 SPAM (MALICIOUS VECTOR DETECTED)")
            st.metric(label="Engine Threat Confidence Rating", value=f"{confidence_percentage:.2f}%")
            st.info("💡 **Security Advisory:** This string pattern contains high-frequency tokens commonly associated with digital phishing variants and bulk advertising sweeps.")
        else:
            st.success(f"### ✅ HAM (SAFE SECURE MESSAGE)")
            st.metric(label="Engine Safety Confidence Rating", value=f"{confidence_percentage:.2f}%")
            st.info("💡 **Security Advisory:** The structural properties of this message indicate standard situational correspondence parameters.")