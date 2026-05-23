import streamlit as st
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered"
)

# Load model
MODEL_PATH = "phishing_model.pkl"

# Title
st.title("🛡️ Phishing Email Detection System")

st.markdown("""
Detect whether an email is **Phishing** or **Safe** using Machine Learning.

This project analyzes:
- Suspicious keywords
- URLs
- Email text patterns
- Security-related content
""")

# Check model existence
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please train the model first.")
    st.stop()

# Load model
model = joblib.load(MODEL_PATH)

# Text area
email_input = st.text_area(
    "Enter Email Content",
    height=250,
    placeholder="Paste suspicious email content here..."
)

# Detection button
if st.button("Detect Email"):

    if email_input.strip() == "":
        st.warning("Please enter email content.")
    else:
        # Prediction
        prediction = model.predict([email_input])[0]
        probability = model.predict_proba([email_input])[0]

        confidence = max(probability) * 100

        st.subheader("Detection Result")

        # Result display
        if prediction == "phishing":
            st.error(f"⚠️ This email is classified as PHISHING")
        else:
            st.success(f"✅ This email is classified as SAFE")

        st.info(f"Confidence Score: {confidence:.2f}%")

        # Probability bars
        st.subheader("Prediction Probabilities")

        phishing_prob = 0
        safe_prob = 0

        if "phishing" in model.classes_:
            phishing_index = list(model.classes_).index("phishing")
            safe_index = list(model.classes_).index("safe")

            phishing_prob = probability[phishing_index] * 100
            safe_prob = probability[safe_index] * 100

        st.progress(int(phishing_prob))
        st.write(f"Phishing Probability: {phishing_prob:.2f}%")

        st.progress(int(safe_prob))
        st.write(f"Safe Probability: {safe_prob:.2f}%")

# Footer
st.markdown("---")
st.markdown(
    "Developed using Python, Scikit-learn, and Streamlit"
)