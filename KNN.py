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
from sklearn.model_selection import cross_val_score
from textblob import TextBlob
import string
import re
import warnings

warnings.filterwarnings('ignore')
nltk.download('words')
nltk.download('stopwords')
print("\n")

tweets_data = pd.read_csv("covid-19_tweets.csv", encoding="utf-8")
print("The Columns in the dataset are\n")
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
    # 4. Remove hashtags
    tweet = re.sub(r'#', '', tweet)

    return tweet

tweets_data['tweet_text'] = tweets_data['tweet_text'].apply(clean_tweet)

#reading the cleaned CSV File
df = pd.read_csv("covid-19_tweets.csv", encoding="iso-8859-1")

print("\n")
print(df.info())

print("\n")
print(df.describe())

print("\n")
#Sample application of sentiment using TextBlob
sentiments = df[['label','tweet_text']]
def detect_sentiment(tweet_text):
    return TextBlob(tweet_text).sentiment.polarity

sentiments["sentiment"]=sentiments["tweet_text"].apply(detect_sentiment)
print(sentiments.head())
print("\n")

def sentiment2(sent):
    if (sent< -0.02):
        return 1
    elif sent>0.02:
        return 3
    else:
        return 2

sentiments["sent"]=sentiments["sentiment"].apply(sentiment2)
print(sentiments.head())

df_neutral=df[df['label']==2]
df_positive=df[df['label']==3]
df_negative=df[df['label']==1]
df1=df[df['label']!=2]

# Drop rows with neutral sentiment as we want to predict for 1 (negative) and 3 (positive)
df = df[df['label'].isin([1, 3])]

#Converting the  data into a format that can be used by ML algorithm
vect = CountVectorizer(lowercase=True, stop_words="english")
x = df.tweet_text
y = df.label
x = vect.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=60)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Classification and Confusion Matrix\n")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("\n")

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_excel('results.xlsx', index=False)

print("Actual vs Predicted\n")
print(results)

cv_score = cross_val_score(knn,x,y,cv=10)
print("Mean of cross validation score", cv_score.mean())

print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#CountPlot for Positive and negative
sns.countplot(x='label', data=df)
plt.title('Distribution of Sentiment Labels')
plt.show()

train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

print("Train accuracy\n",train_accuracy)
print("test accuracy\n",test_accuracy)

unseen_data = pd.read_csv("unseen.csv", encoding="utf-8")
unseen_data['tweet_text'] = unseen_data['tweet_text'].apply(clean_tweet)
X_unseen = vect.transform(unseen_data['tweet_text'])
y_unseen_pred = knn.predict(X_unseen)
unseen_accuracy = accuracy_score(unseen_data['label'], y_unseen_pred)
print("Unseen data accuracy:", unseen_accuracy)
