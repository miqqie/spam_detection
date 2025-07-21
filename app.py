from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from flask import render_template

# Load and preprocess data
df = pd.read_csv("spam_text.csv")
X = df['Message']
Y = df['Category']

# Vectorize
vectorizer = CountVectorizer()
X_CV = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_CV, Y)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate model
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
acc = np.trace(cm) / np.sum(cm)

print("Confusion Matrix:\n", cm)
print("Accuracy without TFIDF is", acc)

# Flask app
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
    
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    return jsonify({'prediction': prediction})

@app.route('/metrics')
def metrics():
    return render_template('metrics.html', 
                           confusion_matrix=cm.tolist(), 
                           accuracy=acc)
