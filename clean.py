import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from textblob import TextBlob
import string
import re
import warnings
warnings.filterwarnings('ignore')


nltk.download('words')
nltk.download('stopwords')

tweets_data = pd.read_csv("covid-19_tweets.csv", encoding="utf-8")

print(tweets_data.head())


def clean_tweet(tweet):
    # Remove @ mentions, URLs, and extra spaces
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    
    # 1. Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # 2. Convert to lowercase
    tweet = tweet.lower()
    # 3. Remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    # 4. Remove hashtags (optional, depending on your analysis)
    tweet = re.sub(r'#', '', tweet)

    return tweet

tweets_data['tweet_text'] = tweets_data['tweet_text'].apply(clean_tweet)

df = pd.read_csv("covid-19_tweets.csv", encoding="iso-8859-1")

print("\n")
print(df.info())

print("\n")
print(df.describe())

df_neutral=df[df['label']==2]
df_positive=df[df['label']==3]
df_negative=df[df['label']==1]
df1=df[df['label']!=2]

# Drop rows with neutral sentiment as we want to predict for 1 (negative) and 3 (positive)
df = df[df['label'].isin([1, 3])]

vect = CountVectorizer(lowercase=True, stop_words="english")
x = df.tweet_text
y = df.label
x = vect.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=60)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted\n")
print(results)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

sns.countplot(x='label', data=df)
plt.title('Distribution of Sentiment Labels')
plt.show()

sentiments = df[['label','tweet_text']]
def detect_sentiment(tweet_text):
    return TextBlob(tweet_text).sentiment.polarity

sentiments["sentiment"]=sentiments["tweet_text"].apply(detect_sentiment)
print(sentiments.head())

def sentiment2(sent):
    if (sent< -0.02):
        return 1
    elif sent>0.02:
        return 3
    else:
        return 2

sentiments["sent"]=sentiments["sentiment"].apply(sentiment2)
print(sentiments.head())

