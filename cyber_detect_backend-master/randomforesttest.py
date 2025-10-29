import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import re
import nltk
from nltk.util import pr
stemmer = nltk.SnowballStemmer('english')
import string
# Try to load NLTK stopwords; fall back to a small built-in list when unavailable/offline
try:
  from nltk.corpus import stopwords
  stopword = set(stopwords.words("english"))
except Exception:
  stopword = set(["the","a","an","and","or","is","are","to","of","in","on","for","with","this","that"])  

df = pd.read_csv("twitter_data.csv")
df['labels'] = df['class'].map({0:"Hate Speech Detected" , 1:"Offensive language detected" , 2: "No hate and offensive speech"})
df = df[['tweet' , 'labels']]

def clean(text):
  text = str(text).lower()
  text = re.sub('\[.*?\]' , '', text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+' , '' , text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '' , text)
  text = re.sub('\n' , '' , text)
  text = re.sub('\w*\d\w*', ''  , text)
  text = [word for word in text.split(' ') if word not in stopword]
  text = " ".join(text)
  text = [stemmer.stem(word) for word in text.split(' ')]
  text = " ".join(text)
  return text

df["tweet"] = df["tweet"].apply(clean)

x = np.array(df["tweet"])
y = np.array(df["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)

def randomforestpredict(model , text):
  test_data = text
  df = cv.transform([test_data]).toarray()
  output = model.predict(df)
  print(output)
  probabilities = model.predict_proba(df)
  print(f"randomforest : {probabilities}")
  return probabilities[0]


def train_randomforest():
  # Train a simple RandomForest model if a pre-trained pickle is unavailable
  X_text = np.array(df["tweet"])  # already cleaned above
  X = cv.transform(X_text).toarray()
  y_labels = np.array(df["labels"])  # use string labels directly
  X_train, X_valid, y_train, y_valid = train_test_split(X, y_labels, test_size=0.1, random_state=42, stratify=y_labels)
  model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
  model.fit(X_train, y_train)
  return model
