from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load and preprocess data
df = pd.read_csv("spam_text.csv")
X = df['Message']
Y = df['Category']
X_CV = CountVectorizer().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_CV, Y)
model = LogisticRegression()
model.fit(X_train, Y_train)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return 'Spam Detection API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    vectorizer = CountVectorizer().fit(X)
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    return jsonify({'prediction': prediction})

