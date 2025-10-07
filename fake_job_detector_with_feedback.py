# fake_job_detector_with_feedback.py
import os
import re
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
import joblib
import streamlit as st
from streamlit_lottie import st_lottie

# ---------------------------
# NLTK Setup (downloads once)
# ---------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Helper functions
# ---------------------------
def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# ---------------------------
# Training with sample weights (feedback has stronger influence)
# ---------------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(base_data_path, feedback_path="user_feedback.csv", feedback_weight=30):
    """
    Train models on base data + feedback.
    feedback_weight: multiplier for feedback examples (makes them more influential).
    Uses sample_weight during model.fit instead of naive duplication.
    """
    # Load base data
    df_base = pd.read_csv(base_data_path)
    df_base = df_base.dropna(subset=["description", "fraudulent"]).reset_index(drop=True)
    df_base["final_text"] = df_base["description"].apply(clean_text)

    feedback_exists = os.path.exists(feedback_path) and os.path.getsize(feedback_path) > 0
    if feedback_exists:
        fb = pd.read_csv(feedback_path)
        fb = fb.dropna(subset=["description", "fraudulent"]).reset_index(drop=True)
        # Ensure label type is integer (0 or 1)
        fb["fraudulent"] = fb["fraudulent"].astype(int)
        fb["final_text"] = fb["description"].apply(clean_text)
        df = pd.concat([df_base, fb], ignore_index=True)
        # create sample weights: base rows weight=1, feedback rows weight=feedback_weight
        w_base = np.ones(len(df_base), dtype=float)
        w_fb = np.ones(len(fb), dtype=float) * float(feedback_weight)
        sample_weights = np.concatenate([w_base, w_fb])
    else:
        df = df_base
        sample_weights = np.ones(len(df), dtype=float)

    X_text = df["final_text"].fillna("")
    y = df["fraudulent"].astype(int).values

    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X_text)

    # train_test_split can split multiple arrays in parallel (including sample_weights)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_tfidf, y, sample_weights, test_size=0.20, random_state=42, stratify=y
    )

    # Logistic Regression (uses sample weights)
    log_model = LogisticRegression(max_iter=500, class_weight="balanced")
    log_model.fit(X_train, y_train, sample_weight=w_train)
    y_pred_log = log_model.predict(X_test)
    acc_log = accuracy_score(y_test, y_pred_log)

    # Random Forest (also supports sample weights)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample")
    rf_model.fit(X_train, y_train, sample_weight=w_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # pick best
    if acc_log >= acc_rf:
        best_model = log_model
        best_name = "Logistic Regression"
    else:
        best_model = rf_model
        best_name = "Random Forest"

    # Save artifacts
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    # Return a bit of diagnostics too
    diagnostics = {
        "acc_log": acc_log,
        "acc_rf": acc_rf,
        "n_base": len(df_base),
        "n_feedback": len(fb) if feedback_exists else 0,
        "feedback_weight": feedback_weight
    }
    return best_model, vectorizer, best_name, diagnostics

# ---------------------------
# Load or train initial model
# ---------------------------
base_dataset = "fake_job_postings.csv"
if not os.path.exists("best_model.pkl"):
    model, vectorizer, best_name, diagnostics = train_model(base_dataset)
else:
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    best_name = "Loaded from disk"
    diagnostics = {"n_base": None, "n_feedback": 0, "acc_log": None, "acc_rf": None, "feedback_weight": None}

# ---------------------------
# Streamlit UI + Styling
# ---------------------------
st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="centered")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #1f1c2c, #928dab, #2C5364, #203A43);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    h1, h2, h3, h4, h5, p, div { color: white !important; }
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.12);
        color: #fff;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: 0.2s;
    }
    .stButton>button:hover { transform: scale(1.03); }
    </style>
""", unsafe_allow_html=True)

# Header animation
st.markdown("""
<div style="text-align:center;">
    <div style="font-size:32px; font-weight:600; color:white;">
        <span class="typing">🕵️ Fake Job Posting Detector</span>
    </div>
</div>
<style>
@keyframes typing { from { width: 0 } to { width: 100% } }
.typing {
  display: inline-block; overflow: hidden; white-space: nowrap;
  border-right: .12em solid orange;
  animation: typing 3.5s steps(30,end), blink-caret .75s step-end infinite;
}
@keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: orange; } }
</style>
""", unsafe_allow_html=True)

# Lottie animation
lottie_job = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_v7f5gn6h.json")
if lottie_job:
    st_lottie(lottie_job, speed=1, height=210, key="detective")

st.write("### Paste a job description below to check if it’s **Real or Fraudulent**.")
user_input = st.text_area("Enter Job Description:", height=200)

# Option to show debug info
show_debug = st.checkbox("Show debug & feedback info (helpful to diagnose retrain behavior)")

# Predict button
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a job description.")
    else:
        with st.spinner("Analyzing job description..."):
            cleaned_text = clean_text(user_input)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]
            # get probability for predicted class
            prob = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vectorized_text)[0]
                prob = probs[prediction]
            else:
                prob = None

        if prediction == 1:
            if prob is not None:
                st.error(f"🚨 This job posting looks **Fraudulent** (confidence: {prob:.2f})")
            else:
                st.error(f"🚨 This job posting looks **Fraudulent**")
        else:
            if prob is not None:
                st.success(f"✅ This job posting looks **Real** (confidence: {prob:.2f})")
            else:
                st.success(f"✅ This job posting looks **Real**")

        # Save for feedback/retrain
        st.session_state["last_input"] = user_input
        st.session_state["last_pred"] = int(prediction)
        if prob is not None:
            st.session_state["last_prob"] = float(prob)

# Feedback section (save user corrections)
if "last_input" in st.session_state:
    st.write("### 🙋 Was this prediction correct?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Correct"):
            fb_df = pd.DataFrame([{
                "description": st.session_state["last_input"],
                "fraudulent": st.session_state["last_pred"]
            }])
            fb_df.to_csv("user_feedback.csv", mode="a", header=not os.path.exists("user_feedback.csv"), index=False)
            st.success("Thanks! Your confirmation was saved.")
    with col2:
        if st.button("👎 Incorrect"):
            # flip label
            true_label = 0 if st.session_state["last_pred"] == 1 else 1
            fb_df = pd.DataFrame([{
                "description": st.session_state["last_input"],
                "fraudulent": true_label
            }])
            fb_df.to_csv("user_feedback.csv", mode="a", header=not os.path.exists("user_feedback.csv"), index=False)
            st.info("Thanks — your correction was saved.")

# Retrain controls
st.write("### 🔁 Retrain Model with Feedback")
col_a, col_b = st.columns([1, 2])
with col_a:
    fb_weight = st.number_input("Feedback weight (higher = feedback influences retrain more)", min_value=1, max_value=500, value=30, step=1)
with col_b:
    if st.button("Retrain Now"):
        if not os.path.exists("user_feedback.csv") or os.path.getsize("user_feedback.csv") == 0:
            st.warning("No feedback found (user_feedback.csv is missing or empty). Add feedback by marking predictions Incorrect/Correct first.")
        else:
            with st.spinner("Retraining model (this may take a moment)..."):
                model, vectorizer, best_name, diagnostics = train_model(base_dataset, feedback_path="user_feedback.csv", feedback_weight=fb_weight)
                # reload saved artifacts for safety
                model = joblib.load("best_model.pkl")
                vectorizer = joblib.load("vectorizer.pkl")

            st.success(f"Model retrained! Chosen model: {best_name}. Diagnostics: {diagnostics}")
            # If we have a last_input, automatically re-run prediction and show new result vs old
            if "last_input" in st.session_state:
                try:
                    cleaned_text = clean_text(st.session_state["last_input"])
                    vec = vectorizer.transform([cleaned_text])
                    new_pred = int(model.predict(vec)[0])
                    new_prob = None
                    if hasattr(model, "predict_proba"):
                        new_prob = model.predict_proba(vec)[0][new_pred]
                    old_pred = st.session_state.get("last_pred", None)
                    old_prob = st.session_state.get("last_prob", None)

                    st.write("#### Comparison (old vs new prediction for your last input):")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Before retrain**")
                        if old_pred is not None:
                            st.write("Label:", "Fraudulent" if old_pred == 1 else "Real")
                        if old_prob is not None:
                            st.write(f"Confidence: {old_prob:.2f}")
                    with col2:
                        st.write("**After retrain**")
                        st.write("Label:", "Fraudulent" if new_pred == 1 else "Real")
                        if new_prob is not None:
                            st.write(f"Confidence: {new_prob:.2f}")

                    # update session with new result (so feedback flows from here on)
                    st.session_state["last_pred"] = new_pred
                    if new_prob is not None:
                        st.session_state["last_prob"] = float(new_prob)
                except Exception as e:
                    st.error(f"Could not re-predict last input after retrain: {e}")

# Show debug info if asked
if show_debug:
    st.write("#### Debug / Feedback Info")
    if os.path.exists("user_feedback.csv") and os.path.getsize("user_feedback.csv") > 0:
        fb = pd.read_csv("user_feedback.csv").dropna(subset=["description", "fraudulent"])
        st.write(f"Feedback rows: {len(fb)}")
        st.write("Feedback label distribution:")
        st.write(fb["fraudulent"].value_counts().to_dict())
        st.write("Sample feedback rows (last 5):")
        st.dataframe(fb.tail(5)[["description", "fraudulent"]])
    else:
        st.write("No feedback file found yet (user_feedback.csv missing or empty).")

    # Show which model loaded
    st.write("Model loaded:", best_name)
    if diagnostics:
        st.write("Last diagnostic:", diagnostics)

st.write("---")
st.caption("Tip: If a single feedback doesn't flip the prediction, try increasing 'Feedback weight' temporarily or add a couple more feedback corrections of similar examples. Very short texts (like 'Earn 500k per month') may still be ambiguous — adding a sentence or two of context helps the model generalize.")
