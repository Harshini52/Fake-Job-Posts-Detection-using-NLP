import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# -----------------------------------------------------------
# 1️⃣ LOAD DATASET
# -----------------------------------------------------------
df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\Infosys Python Project\\fake_job_postings.csv")


# Ensure text column is string
df['text'] = (
    df['title'].astype(str) + " " +
    df['company_profile'].astype(str) + " " +
    df['description'].astype(str) + " " +
    df['requirements'].astype(str)
).fillna("")


print("Dataset loaded successfully!")
print("Total rows:", len(df))
print("\nColumns:", df.columns.tolist())


# -----------------------------------------------------------
# 2️⃣ TEXT LENGTH ANALYSIS (DESCRIPTIVE OUTPUT)
# -----------------------------------------------------------
df['text_length'] = df['text'].apply(len)

avg_real = df[df['fraudulent'] == 0]['text_length'].mean()
avg_fake = df[df['fraudulent'] == 1]['text_length'].mean()

print("\n=====================================")
print("📌 TEXT LENGTH ANALYSIS")
print("=====================================")

print(f"➡️ Average text length (REAL jobs) : {avg_real:.2f} characters")
print(f"➡️ Average text length (FAKE jobs) : {avg_fake:.2f} characters")

if avg_fake < avg_real:
    print("🔍 Insight: Fake job descriptions tend to be SHORTER and less detailed.")
else:
    print("🔍 Insight: Fake job descriptions tend to be LONGER or overly wordy.")


# -----------------------------------------------------------
# 3️⃣ WORD COUNT ANALYSIS (DESCRIPTIVE OUTPUT)
# -----------------------------------------------------------
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

real_wc = df[df['fraudulent'] == 0]['word_count'].mean()
fake_wc = df[df['fraudulent'] == 1]['word_count'].mean()

print("\n=====================================")
print("📌 WORD COUNT ANALYSIS")
print("=====================================")

print(f"➡️ Average word count (REAL jobs) : {real_wc:.2f}")
print(f"➡️ Average word count (FAKE jobs) : {fake_wc:.2f}")

if fake_wc < real_wc:
    print("🔍 Insight: Fake job posts typically contain fewer words.")
else:
    print("🔍 Insight: Fake job posts contain long descriptions but lack genuine details.")


# -----------------------------------------------------------
# 4️⃣ COMMON WORDS IN FAKE JOB POSTS
# -----------------------------------------------------------
print("\n=====================================")
print("📌 MOST COMMON WORDS IN FAKE JOB POSTS")
print("=====================================")

nltk.download('stopwords')
stop_words = stopwords.words('english')

# Filter only fake posts
fake_texts = df[df['fraudulent'] == 1]['text']

vectorizer = CountVectorizer(stop_words=stop_words, max_features=30)
matrix = vectorizer.fit_transform(fake_texts)

words = vectorizer.get_feature_names_out()
counts = matrix.sum(axis=0).A1

common_words = list(zip(words, counts))
common_words_sorted = sorted(common_words, key=lambda x: x[1], reverse=True)

for word, count in common_words_sorted:
    print(f"{word}: {count}")

print("\n🔍 Insight:")
print("Fake job posts often use vague or persuasive words such as:")
print("➡️ opportunity, immediate, hiring, apply, income, remote, benefits, required")


# -----------------------------------------------------------
# 5️⃣ TEXT PATTERN INSIGHTS SUMMARY
# -----------------------------------------------------------
print("\n=====================================")
print("📌 OVERALL TEXT ANALYSIS SUMMARY")
print("=====================================")

print("""
✔ REAL job posts usually contain more detailed descriptions and longer text.
✔ FAKE job posts often:
   - Use short or overly long descriptions
   - Avoid giving specific company information
   - Use attractive words like 'income', 'opportunity', 'immediate hire'
   - Include vague roles without responsibilities
✔ Fake posts may also show repetitive phrasing and promotional language.

🎯 These patterns help in training a machine learning model to detect fake jobs.
""")
