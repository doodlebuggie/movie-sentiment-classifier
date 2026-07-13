import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def load_split_data():
    with open('data/data.pkl', 'rb') as f: unpickled_data = pickle.load(f)

    X_train = unpickled_data["X_train"]
    X_test = unpickled_data["X_test"]
    y_train = unpickled_data["y_train"]
    y_test = unpickled_data["y_test"]
    vectorizer = unpickled_data["vectorizer"]

    return X_train, X_test, y_train, y_test, vectorizer

X_train, X_test, y_train, y_test, vectorizer = load_split_data()
print("Data loaded")

def train_model(X_train, y_train):
    classifier = MultinomialNB()
    trained_model = classifier.fit(X_train, y_train)
    return trained_model

model = train_model(X_train, y_train)
print("model traind")

def evaluate_model(model, X_test, y_test):
    y_prediction = model.predict(X_test)

    # accuracy
    accuracy = accuracy_score(y_test, y_prediction)
    # precision
    precision = precision_score(y_test, y_prediction, average = 'macro')
    # recall
    recall = recall_score(y_test, y_prediction, average = 'macro')
    # confusion matrix
    cm = confusion_matrix(y_test, y_prediction)

    print("accuracy ", accuracy, "precision", precision, "recall", recall)
    print("confusion_matrix\n", cm)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "confusion_matrix": cm}

evaluate_model(model, X_test, y_test)
print("model evaluated")

#from AI
def show_top_words(model, vectorizer, n=10):
    words = vectorizer.get_feature_names_out()

    neg_index = list(model.classes_).index('neg')
    pos_index = list(model.classes_).index('pos')
    
    neg_scores = model.feature_log_prob_[neg_index]
    pos_scores = model.feature_log_prob_[pos_index]
   
    diff_scores = pos_scores - neg_scores

    word_scores = list(zip(words, diff_scores))
    word_scores_sorted = sorted(word_scores, key=lambda pair: pair[1])

    print("Top negative words:")
    for word, score in word_scores_sorted[:n]:
        print(word, score)

    print("\nTop positive words:")
    for word, score in word_scores_sorted[-n:]:
        print(word, score)

show_top_words(model, vectorizer)

def save_model(model, vectorizer):
    trained_data = {
        "model": model,
        "vectorizer": vectorizer
    }

    with open("data/trained_data.pkl", "wb") as file:
        pickle.dump(trained_data, file)

save_model(model, vectorizer)
print("model saved")
print("DONE!")