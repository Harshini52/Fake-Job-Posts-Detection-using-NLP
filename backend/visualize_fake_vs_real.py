import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\Infosys Python Project\\fake_job_postings.csv")

# fraudulent = 0 (real), 1 (fake)
label_counts = df['fraudulent'].value_counts()

plt.figure(figsize=(6,5))
plt.bar(label_counts.index, label_counts.values, color=['green','red'])
plt.xticks([0,1], ['Real', 'Fake'])
plt.xlabel("Job Post Type")
plt.ylabel("Count")
plt.title("Distribution of Fake vs Real Job Posts")
plt.show()
