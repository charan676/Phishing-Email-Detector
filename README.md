# 🛡️ Phishing Email Detector

A Machine Learning based cybersecurity project that detects phishing emails using NLP and Scikit-learn. The system analyzes suspicious keywords, URLs, and email patterns to classify emails as Phishing or Safe.

---

## 🚀 Features

- Phishing Email Detection
- URL & Keyword Analysis
- TF-IDF Text Vectorization
- Random Forest Classification
- Streamlit Web Interface
- Accuracy & Confusion Matrix Visualization
- Real-time Prediction System

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 📂 Project Structure

Phishing-Email-Detector/
│
├── phishing_detector.py
├── streamlit_app.py
├── phishing_model.pkl
├── requirements.txt
├── README.md
│
├── data/
│   └── emails.csv
│
└── output/
    └── confusion_matrix.png

---

## ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt

---

## ▶️ Train the Model

```bash
python phishing_detector.py --mode train
```

---

## 🌐 Run Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Example Phishing Email

```text
URGENT! Verify your bank account immediately at http://fakebank.com
```

---

## 📊 Output

- Email Classification
- Confidence Score
- Confusion Matrix
- Probability Visualization

---

## 👨‍💻 Author

RamCharan Narkedimilli
