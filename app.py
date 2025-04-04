from flask import Flask, request, jsonify, render_template
import pickle
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

app = Flask(__name__)

# Load saved model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Function to preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

@app.route("/")
def home():
    return render_template("index.html")  # Serve frontend

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    news_text = data["text"]

    # Preprocess and vectorize text
    processed_text = preprocess_text(news_text)
    vectorized_text = vectorizer.transform([processed_text])

    # Predict
    prediction = model.predict(vectorized_text)[0]
    result = "Real News ✅" if prediction == 1 else "Fake News ❌"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    # Simulated prediction
    prediction = "Fake" if "clickbait" in text.lower() else "Real"
    
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
