import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv('otp_samples_1000.csv')
X = data['message']
y = data['label']

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
model.fit(X, y)

joblib.dump(model, 'otp_spoof_model.pkl')
print("✅ Model trained and saved.")