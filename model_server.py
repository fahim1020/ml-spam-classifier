from flask import Flask, request, jsonify
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure necessary NLTK data is downloaded
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')

# Load the model and vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Initialize PorterStemmer
ps = PorterStemmer()

# Text preprocessing function
def text_transform(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    # Remove stopwords and punctuation, apply stemming
    filtered_words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words("english")]

    return " ".join(filtered_words)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("text", "")  # Get text from request
    
    if not data:
        return jsonify({"error": "No text provided"}), 400  # Handle empty requests
    
    # Apply text transformation
    transformed_text = text_transform(data)
    
    # Convert text to vector format
    vectorized_text = vectorizer.transform([transformed_text])

    # Get prediction
    prediction = model.predict(vectorized_text)[0]
    if prediction == 0:
        predict_label = "Not Spam"
    else:
        predict_label = "Spam"


    return jsonify({"prediction": predict_label})  

if __name__ == "__main__":
    app.run(debug=True, port=5000)  
