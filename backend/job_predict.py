import joblib
import numpy as np
from scipy.sparse import hstack
from scam_keywords import scam_keywords   # ✅ correct import

# Load trained model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def count_scam_keywords(text):
    text = text.lower()
    return sum(kw in text for kw in scam_keywords)

def predict_job_post(text):
    text = text.lower()

    # TF-IDF features
    text_vec = vectorizer.transform([text])

    # Scam keyword count
    scam_score = np.array([[count_scam_keywords(text)]])

    # Combine features (same as training)
    final_input = hstack([text_vec, scam_score])

    prediction = model.predict(final_input)[0]

    return "FAKE JOB ❌" if prediction == 1 else "REAL JOB ✅"
