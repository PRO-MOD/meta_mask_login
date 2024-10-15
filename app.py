from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('best_model.h5')

# Define the tokenizer and stemmer for preprocessing
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")

# Define sequence length (make sure this matches your model's expected input)
sequence_length = 100  # Change this as per your model requirements

# Preprocessing function to tokenize and stem input text
def preprocess_text(text):
    words = tokenizer.tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

# Route for the homepage
@app.route('/success')
def home():
    return render_template('success.html')

# Route to handle form submission and predict the class
@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from the form
    input_text = request.form.get('text')

    # Preprocess the input text
    processed_text = preprocess_text(input_text)
    
    # Tokenize and pad the processed input
    tokenizer1 = Tokenizer()
    tokenizer1.fit_on_texts([processed_text])
    sequences = tokenizer1.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=sequence_length)

    # Make predictions
    predictions = model.predict(np.array(padded_sequences))
    
    # Convert predictions to class labels
    predicted_label = 1 if predictions[0] >= 0.5 else 0
    class_mapping = {0: "fake", 1: "true"}
    predicted_class_name = class_mapping[predicted_label]

    # Return the predicted result to the frontend
    return jsonify({"prediction": predicted_class_name})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
