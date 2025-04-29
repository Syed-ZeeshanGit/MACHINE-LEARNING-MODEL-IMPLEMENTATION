import joblib

# Load model and vectorizer
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict_spam(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return "SPAM" if prediction[0] == 1 else "HAM"

# Example
if __name__ == "__main__":
    test_text = "Congratulations! You've won a free iPhone. Click here to claim now!"
    print("Prediction:", predict_spam(test_text))
