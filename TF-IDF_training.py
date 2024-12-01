# Importing necessary libraries
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB

# Loading spacy's small english model
nlp = spacy.load("en_core_web_sm")


# Defining a function to lemmatize and remove stopwords from the text
def text_preprocessing(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


# Loading the dataset from kaggle
df = pd.read_csv("datasets/SMS Spam Collection Dataset/spam.csv", encoding='ISO-8859-1')

df["spam"] = df["v1"].map({"ham": 0, "spam": 1})


# select n rows for training and testing
df_train = df #.head(10000)


# Extract messages and labels
messages = df['v2']
spams = df['spam']

# Fill NaN values in 'text' column with empty strings
messages = messages.fillna('')

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(messages, spams, test_size=0.2, random_state=42)


# Looping over the classifiers and evaluating their performance
# Creating a pipeline with the classifier
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(preprocessor=text_preprocessing)),
    # ("vectorization", CountVectorizer(preprocessor=text_preprocessing)),
    # ("clf", SGDClassifier())
    ("clf", SVC())
])
# Fitting the pipeline on the training data
pipeline.fit(X_train, y_train)
# Evaluating the pipeline on the testing data
y_pred = pipeline.predict(X_test)
print("Accuracy ", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("--------------------------------")


while 1:
    user_input = input("Enter a text message for classification: ")
    preprocessed_input = text_preprocessing(user_input)
    user_prediction = pipeline.predict([preprocessed_input])
    print("Predicted spam:", "spam" if user_prediction[0] == 1 else "ham")
