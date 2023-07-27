import os
import csv
import re
import chardet
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import tkinter as tk
from tkinter import ttk

def preprocess_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Remove special characters
    text = text.lower()  # Convert text to lowercase
    return text

def load_data(filename):
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    data = []
    with open(filename, encoding=encoding) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            polarity = int(row[0])
            if polarity == 0:
                sentiment = 'negative'
            elif polarity == 2:
                sentiment = 'neutral'
            else:
                sentiment = 'positive'
            text = preprocess_text(row[5])
            data.append({'text': text, 'sentiment': sentiment})
    return data

def classify_subjectivity(text):
    blob = TextBlob(text)
    if blob.sentiment.subjectivity > 0.7:
        return 'very subjective'
    elif blob.sentiment.subjectivity > 0.5:
        return 'somewhat subjective'
    else:
        return 'objective'

# Create the pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(solver='liblinear', random_state=42)),
])

# Define the model file name
model_file = 'text_clf.pkl'

if not os.path.exists(model_file):
    # Train the model if it hasn't been trained yet
    train_data = load_data(r'C:\Users\shinj\Documents\Python Jupyter\trainingandtestdata\training.1600000.processed.noemoticon.csv')
    X_train = [d['text'] for d in train_data]
    y_train = [d['sentiment'] for d in train_data]
    text_clf.fit(X_train, y_train)

    # Save the trained model
    with open(model_file, 'wb') as f:
        pickle.dump(text_clf, f)

# Load the saved model
with open(model_file, 'rb') as f:
    text_clf = pickle.load(f)

def predict_sentiment_subjectivity(text):
    sentiment = text_clf.predict([text])[0]
    subjectivity = classify_subjectivity(text)
    return sentiment, subjectivity

def submit_text():
    user_input = text_entry.get()
    predicted_sentiment, predicted_subjectivity = predict_sentiment_subjectivity(user_input)
    result.set(f"The sentiment analysis indicates that you are feeling {predicted_sentiment} and your thoughts are {predicted_subjectivity}.")

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis")

# Create a label for the input field
text_label = ttk.Label(root, text="Please enter your thoughts:")
text_label.pack(padx=10, pady=10)

text_entry = ttk.Entry(root, width=50)
text_entry.pack(padx=10, pady=10)

# Create a submit button
submit_button = ttk.Button(root, text="Analyze", command=submit_text)
submit_button.pack(padx=10, pady=10)

# Create a label to display the result
result = tk.StringVar()
result_label = ttk.Label(root, textvariable=result)
result_label.pack(padx=10, pady=10)

# Run the main loop to display the window
root.mainloop()
