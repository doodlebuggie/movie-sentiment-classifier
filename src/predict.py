import pickle
import string 
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def load_trained_data():
    with open ("data/trained_data.pkl", "rb") as file: 
        unpickle_trained_data = pickle.load(file)

    model = unpickle_trained_data["model"]
    vectorizer = unpickle_trained_data["vectorizer"]

    return model, vectorizer

model, vectorizer = load_trained_data()
print("trained data loaded")

def predict_sentiment(text, model, vectorizer):
    cleaned_text = re.sub(f"[{string.punctuation}]",'', text).strip().lower()
    transformed_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_input)

    return prediction[0]

reviews = ["this movie was great!", "This movie sucked", "The movie was okay", "I barely got through the movie"]

for review in reviews:
    prediction = predict_sentiment(review, model, vectorizer)
    print(review, ":", prediction)

print("Done")