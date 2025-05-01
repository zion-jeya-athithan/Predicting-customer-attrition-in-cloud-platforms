# retention_offer_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import random
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from gtts import gTTS
import os
import speech_recognition as sr

# Load and preprocess dataset
try:
    df = pd.read_excel("customer_data.xlsx")
except FileNotFoundError:
    st.error("❌ Dataset file 'customer_data.xlsx' not found. Please upload it.")
    st.stop()

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Add noise columns (optional realism)
df['noise_1'] = np.random.rand(len(df))
df['noise_2'] = np.random.randint(0, 100, len(df))

# Label churn probabilistically
def label_churn(row):
    prob = 0
    if row['login_frequency'] < 3:
        prob += 0.4
    if row['outstanding_balance'] > 350:
        prob += 0.3
    if row['satisfaction_ratings'] < 2:
        prob += 0.3
    return 1 if random.random() < prob else 0

df['churn_status'] = df.apply(label_churn, axis=1)

# Simulate Retention Offers
def simulate_offer(row):
    if row['login_frequency'] < 2 and row['satisfaction_ratings'] < 2:
        return "10% Discount"
    elif row['outstanding_balance'] > 400:
        return "Flexible Payment Plan"
    elif row['satisfaction_ratings'] < 3:
        return "Free Support Call"
    else:
        return "No Offer"

df['recommended_offer'] = df.apply(simulate_offer, axis=1)

# Encode offer labels
offer_encoder = LabelEncoder()
df['offer_encoded'] = offer_encoder.fit_transform(df['recommended_offer'])

# Model training for offer recommendation
features = ['login_frequency', 'outstanding_balance', 'satisfaction_ratings']
X = df[features]
y = df['offer_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
offer_model = RandomForestClassifier()
offer_model.fit(X_train, y_train)

# RL-based offer predictor (simulated for now)
def predict_best_offer_rl(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    pred_offer_code = offer_model.predict(input_df)[0]
    return offer_encoder.inverse_transform([pred_offer_code])[0]

# Explainability function
def explain_offer_reason(input_data):
    login_freq, outstanding_bal, satisfaction = input_data
    reasons = []
    if login_freq < 3:
        reasons.append("low login frequency (customer inactivity)")
    if outstanding_bal > 350:
        reasons.append("high outstanding balance (financial risk)")
    if satisfaction < 3:
        reasons.append("low satisfaction rating (disengagement risk)")
    if not reasons:
        reasons.append("overall stable engagement metrics")
    explanation = "Offer recommended because: " + ", ".join(reasons) + "."
    return explanation

# 🎮 Gamification Engine
def calculate_gamified_score(login_freq, outstanding_bal, satisfaction, churn_status):
    score = 0
    if login_freq >= 5:
        score += 10
    if satisfaction >= 4:
        score += 10
    if outstanding_bal <= 350:
        score += 10
    if churn_status == 0:
        score += 10
    return score

def assign_gamified_badge(score):
    if score <= 20:
        return "🐢 Basic User", "Bronze"
    elif 21 <= score <= 30:
        return "⚡ Engaged User", "Silver"
    elif 31 <= score <= 40:
        return "🌟 Power User", "Gold"
    else:
        return "🔥 Loyalty Champion", "Platinum"

# Streamlit Dashboard
st.set_page_config(page_title="AI Retention Offer Recommender", layout="centered")
st.title(" ChurnShield AI")
st.markdown("---")
st.subheader("Customer Profile Input")
login_freq = st.slider("Login Frequency", 0, 10, 3)
outstanding_bal = st.slider("Outstanding Balance", 0, 1000, 250)
satisfaction = st.slider("Satisfaction Ratings", 1, 5, 3)

if st.button(" Recommend Offers"):
    input_data = [login_freq, outstanding_bal, satisfaction]
    offer_label_rl = predict_best_offer_rl(input_data)
    explanation_text = explain_offer_reason(input_data)

    st.success(f"📌 RL-based Recommended Offer: **{offer_label_rl}**")
    st.info(explanation_text)

    sample_churn_status = 0
    total_score = calculate_gamified_score(login_freq, outstanding_bal, satisfaction, sample_churn_status)
    badge, level = assign_gamified_badge(total_score)

    

    st.markdown("### 🔍 Feature Impact Summary")
    impact_df = pd.DataFrame({
        "Feature": ["Login Frequency", "Outstanding Balance", "Satisfaction Rating"],
        "Input Value": [login_freq, outstanding_bal, satisfaction],
        "Threshold (Risk Level)": ["< 3", "> 350", "< 3"],
        "Risk Flagged": [
            "⚠️ Yes" if login_freq < 2 else "✅ No",
            "⚠️ Yes" if outstanding_bal > 350 else "✅ No",
            "⚠️ Yes" if satisfaction < 2 else "✅ No"
        ]
    })
    st.table(impact_df)

    st.markdown("### SPEAKATRON")
    lang_choice = st.selectbox("", ["English", "Hindi", "Spanish", "French", "German"])
    lang_dict = {"English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}
    lang_code = lang_dict.get(lang_choice, "en")
    lang_code = lang_dict.get(lang_choice, "hi")
    lang_code = lang_dict.get(lang_choice, "es")
    lang_code = lang_dict.get(lang_choice, "fr")

    try:
        tts = gTTS(text=explanation_text, lang=lang_code)
        tts.save("explanation.mp3")
        audio_file = open("explanation.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")
    except Exception as e:
        st.error(f"Voice generation failed: {e}")

   

st.markdown("---")
st.subheader("📊 Retention Offer Distribution")
st.bar_chart(df['recommended_offer'].value_counts())

st.markdown("---")
st.caption("Built with ❤️ for Customer Retention AI Dashboard")