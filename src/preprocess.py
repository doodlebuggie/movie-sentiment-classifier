import pandas as pd
import re
import string
import pickle
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("STARTING")
def load_dataset():
    data_reviews = []
    for category in movie_reviews.categories(): #goes through pos then neg
        for fileid in movie_reviews.fileids(category): # goes through every file in that cat
            text = movie_reviews.raw(fileid)
            data_reviews.append({"label": category, "text": text})
    df = pd.DataFrame(data_reviews)    
    return df

def clean_text(text):
    if isinstance(text, str): #isinstance() is a built-in Python function used to check if an object belongs to a specific data type (like a string, integer, or list)
        return re.sub(f"[{string.punctuation}]",'', text).strip().lower()
    return text

df = load_dataset()
print("dataset loaded")
df['text'] = df['text'].apply(clean_text)
print("text cleaned")

def vectorize_text(cleaned_text):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_text)
    
    return X, vectorizer

X, vectorizer = vectorize_text(df['text'])

def split_into_test_train(X, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_into_test_train(X, df['label'])

print("data sucessfully splitted")

def save_vars(X_train, X_test, y_train, y_test, vectorizer):
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "vectorizer": vectorizer
    }

    with open("data/data.pkl", "wb") as file:
        pickle.dump(data, file)

save_vars(X_train, X_test, y_train, y_test, vectorizer)

print("variables saved using pickle!")
print("DONE")