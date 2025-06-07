import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("Tweets.csv")
print("Jumlah data:", len(df))
print("Label sentimen unik:", df['airline_sentiment'].unique())

# 2. Ambil kolom teks & label
X_raw = df['text']
y = df['airline_sentiment']

# 3. Text cleaning (basic preprocessing)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # hapus URL
    text = re.sub(r"@\w+", "", text)     # hapus mention
    text = re.sub(r"#\w+", "", text)     # hapus hashtag
    text = re.sub(r"[^\w\s]", "", text)  # hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

X_clean = X_raw.apply(clean_text)

# 4. Ubah teks ke fitur numerik (CountVectorizer)
vectorizer = CountVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X_clean)

# 5. Split data (train/test 80:20)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 6. Training model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Prediksi dan evaluasi
y_pred = model.predict(X_test)

print("\n===== EVALUASI MODEL =====")
print("Akurasi: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Visualize Confusion Matrix
labels = model.classes_
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Naive Bayes Twitter Sentiment")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()