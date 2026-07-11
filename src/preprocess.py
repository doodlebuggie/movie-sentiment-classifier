"""
1. Load your data — either re-run the NLTK loop from Day 1, or read your saved CSV if you have one (pd.read_csv(...))
2. Clean the text — lowercase everything, strip out punctuation/weird characters. This is usually a small function you write and then apply to every row.
3. Turn text into numbers (TF-IDF) — this is the sklearn part. You'll import TfidfVectorizer from sklearn.feature_extraction.text, and it takes your cleaned text column and converts it into a big matrix of numbers, one row per review, one column per word (weighted by importance).
4. Split into train and test sets — train_test_split from sklearn.model_selection, splitting both your TF-IDF numbers and your labels together, so you have a training chunk (model learns from this) and a testing chunk (you evaluate on this later, untouched during training).
5. Save the results — so 03_train.py doesn't have to redo all this work; you'd typically save the split data using something like pickle or joblib.

"""

print("IMPORTING...")
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