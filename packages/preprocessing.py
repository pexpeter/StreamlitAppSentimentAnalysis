import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_data():
    
    df = pd.read_csv("data/tweets.csv",
                encoding='ISO-8859-1', 
                header = None, 
                names= ["target", "tweet_id", "date", "flag", "user", "tweet"]
                )
    df = df.drop(columns = ["tweet_id", "date", "flag", "user"])
    
    df["target"] = df["target"].map({0:0,4:1})
    
    def text_processing(text):
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+', '',text)
        text = re.sub(r'#\w+', '',text )
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text.strip()

    df["tweet"] = df["tweet"].apply(text_processing)
    
    return df

def model_feat():
    df = load_data()
    X = df["tweet"]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vector = CountVectorizer(stop_words='english')
    X_train_vector = vector.fit_transform(X_train)
    return X_train_vector, y_train, vector
