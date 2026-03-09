import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. הכנת נתונים (Data Preparation)
# כאן אנחנו יוצרים דאטה-סט בסיסי. במציאות, היינו טוענים קובץ CSV גדול.
data = {
    'text': [
        'I love this product, it is amazing', 'Best purchase ever', 'Great quality and fast delivery',
        'I am very happy with this', 'Excellent service', 'Truly wonderful experience',
        'I hate this, it is terrible', 'Worst experience ever', 'Poor quality and slow delivery',
        'I am very disappointed', 'Horrible service', 'Waste of money', 'Never buying again'
    ],
    'label': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0] # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# 2. חלוקת הנתונים לאימון ובדיקה (Train-Test Split)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 3. הפיכת טקסט למספרים (Vectorization)
# המחשב לא מבין מילים, אז אנחנו הופכים כל משפט לוקטור של מספרים
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 4. אימון המודל (Model Training)
# נשתמש ב-Naive Bayes, בדיוק כמו בפרויקט ה-CKD שלך - זה מראה על עקביות!
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# 5. בדיקת ביצועים (Evaluation)
predictions = model.predict(X_test_counts)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100}%")
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 6. בדיקה על משפט חדש (Inference)
def predict_sentiment(new_text):
    new_text_counts = vectorizer.transform([new_text])
    prediction = model.predict(new_text_counts)
    return "Positive" if prediction[0] == 1 else "Negative"

# נסה בעצמך:
print(f"Prediction for 'This is great': {predict_sentiment('This is great')}")
print(f"Prediction for 'This is bad': {predict_sentiment('This is bad')}")