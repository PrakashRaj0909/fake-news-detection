import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not available

nltk.download('stopwords')

# Load datasets
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

# Add labels
true_news['label'] = 1  # Real news
fake_news['label'] = 0  # Fake news

# Combine datasets
news_data = pd.concat([true_news, fake_news], axis=0).reset_index(drop=True)

# Shuffle the data
news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    stop_words = set(stopwords.words('english'))  # Get stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Apply preprocessing
news_data['text'] = news_data['text'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(news_data['text'], news_data['label'], test_size=0.2, random_state=7)

# Convert text data into numerical vectors using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize and train Logistic Regression modelnew
model = LogisticRegression()
model.fit(tfidf_train, y_train)

# Make predictions
y_pred = model.predict(tfidf_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)

# Function to predict if news is fake or real
def predict_news(news_text):
    processed_text = preprocess_text(news_text)  # Preprocess input
    vectorized_text = tfidf_vectorizer.transform([processed_text])  # Convert to TF-IDF
    prediction = model.predict(vectorized_text)  # Predict using trained model
    return "Real News ✅" if prediction[0] == 1 else "Fake News ❌"

# Get user input and classify
user_news = input("Enter the news article: ")
result = predict_news(user_news)
print("\nPrediction:", result)

import pickle

# Save trained model
with open("fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")
