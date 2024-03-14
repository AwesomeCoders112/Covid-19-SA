import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download necessary NLTK data
nltk.download('stopwords')

# Load the dataset
tweets_data = pd.read_csv("covid-19_tweets.csv", encoding="utf-8")

# Define a function to clean tweets
def clean_tweet(tweet):
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    tweet = re.sub(r'#', '', tweet)
    return tweet

# Apply the cleaning function to the 'tweet_text' column
tweets_data['tweet_text'] = tweets_data['tweet_text'].apply(clean_tweet)

# Vectorize the text data
vect = CountVectorizer(lowercase=True, stop_words="english")
x = vect.fit_transform(tweets_data['tweet_text'])
y = tweets_data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Predict the sentiment labels for the test set
y_pred = nb.predict(X_test)

# Print the classification report and accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Create a DataFrame for the results
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the distribution of sentiment labels
sns.countplot(x='Predicted', data=results)
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'])
plt.show()