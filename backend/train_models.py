import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack

# ============================================================
# 1. Load Dataset
# ============================================================
df = pd.read_csv("fake_job_postings.csv")

# Combine all text fields into one column
df['text'] = (
    df['title'].astype(str) + " " +
    df['company_profile'].astype(str) + " " +
    df['description'].astype(str) + " " +
    df['requirements'].astype(str) + " " +
    df['benefits'].astype(str)
).fillna("")

y = df['fraudulent']

# ============================================================
# 2. Balance Dataset
# ============================================================
df_majority = df[df.fraudulent == 0]
df_minority = df[df.fraudulent == 1]

df_minority_up = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_up])

X = df_balanced["text"]
y = df_balanced["fraudulent"]

print("Balanced dataset counts:")
print(y.value_counts())

# ============================================================
# 3. Scam Keyword List
# ============================================================
scam_keywords = [
    # 💰 Payment / Fee Related
    "registration fee", "processing fee", "refundable fee", "training fee",
    "security deposit", "initial payment", "pay some amount",
    "pay before joining", "deposit money", "send money",
    "payment required", "pay reg fee", "registration charges",
    "one time fee", "pay immediately",

    # 🪪 Personal Data Misuse
    "send aadhar", "send id proof", "send documents",
    "send pan card", "send resume", "drop your cv", "send cv",

    # 📧 Unofficial Communication
    "send to gmail", "gmail.com", "@gmail.com",
    "whatsapp interview", "telegram contact",
    "call hr", "phone number provided",

    # 🚨 Urgency & Pressure
    "urgent hiring", "immediate hiring", "immediate joining",
    "instant joining", "apply immediately", "limited seats",
    "only today", "hiring !!!", "we are hiring",

    # ❌ No Interview / Auto Selection
    "no interview", "no technical test",
    "resume verification only", "selected without interview",
    "direct selection", "auto offer", "instant offer",

    # 💸 Unrealistic Income Promises
    "guaranteed income", "earn per day", "earn money",
    "daily payment", "weekly payout", "salary in hand",
    "fixed salary", "high salary", "earn from home",
    "35k to 58k", "25,000 per month",

    # 🔗 Malicious Links
    "click the link", "open the link",
    "google form link", "drive link",

    # 📄 Fake Offer Letter Language
    "sending offer letter", "offer letter mailed",
    "offer letter", "attached offer letter",
    "offer letter pdf", "attached pdf",
    "review the attached pdf",
    "offer id", "offer letter id",

    # ✅ Forced Acceptance
    "accept offer", "accept offer button",
    "click accept", "confirm offer",
    "confirm acceptance",

    # 🎓 Internship & PPO Scams
    "internship unpaid", "stipend unpaid",
    "unpaid internship", "remote unpaid internship",
    "internship certificate", "pre placement offer",
    "ppo", "ppo opportunity", "ppo upto",

    # 🧳 Fake Perks & Gifts
    "welcome kit", "gift hamper",
    "bag water bottle",
    "delivered to your home address",
    "laptop kit will be provided",
    "device delivered to home",

    # 🏠 Work From Home Bait
    "work from home", "wfh",
    "online mode", "remote job",
    "remote internship", "easy work",

    # 👥 Over-Broad Eligibility
    "any graduate", "college students freshers",
    "no experience required", "freshers eligible",
    "all degrees accepted",

    # 🏢 Fake Credibility Signals
    "iso certified company", "certified company",
    "verified company", "hr head",
    "talent acquisition",

    # 👍 Social Media Engagement Traps
    "like this post", "hit the like button",
    "drop a comment", "comment your email id",
    "comment email", "share this post",
    "repost", "profile will be shortlisted",
    "shortlisted now",

    # 🧾 Fake Onboarding Promises
    "onboarding details will be shared",
    "login system will be provided",
    "workspace access later"
]




def count_scam_keywords(text):
    text = text.lower()
    return sum(kw in text for kw in scam_keywords)

# ============================================================
# 4. Train-Test Split (split FIRST)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 5. Compute scam_score AFTER split (important!)
# ============================================================
scam_train = X_train.apply(count_scam_keywords).values.reshape(-1, 1)
scam_test = X_test.apply(count_scam_keywords).values.reshape(-1, 1)

# ============================================================
# 6. TF-IDF Vectorizer
# ============================================================
# ============================================================
# 6. TF-IDF Vectorizer
# ============================================================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔴 SAVE TF-IDF VECTORIZER
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


# ============================================================
# 7. Combine TF-IDF + scam_score
# ============================================================
X_train_final = hstack([X_train_vec, scam_train])
X_test_final = hstack([X_test_vec, scam_test])

# ============================================================
# 8. Train SVM Model
# ============================================================
svm_model = LinearSVC()
svm_model.fit(X_train_final, y_train)
joblib.dump(svm_model, "svm_model.pkl")

# ============================================================
# 9. Train Logistic Regression Model
# ============================================================
log_model = LogisticRegression(max_iter=3000)
log_model.fit(X_train_final, y_train)
joblib.dump(log_model, "logistic_model.pkl")

# ============================================================
# 10. Evaluate Models
# ============================================================
print("\n🔥 SVM RESULTS:")
svm_pred = svm_model.predict(X_test_final)
print(classification_report(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))

print("\n🎯 Logistic Regression RESULTS:")
log_pred = log_model.predict(X_test_final)
print(classification_report(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))

print("\n✅ Training completed successfully!")
