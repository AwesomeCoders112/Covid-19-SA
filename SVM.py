import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources (if not already done)
import nltk
nltk.download('stopwords')

# Load your dataset
# Assuming 'your_dataset.csv' contains 'label' and 'tweet_text' columns
df = pd.read_csv('covid-19_tweets.csv')

# Map numerical labels to sentiment labels
sentiment_mapping = {1: 'negative', 2: 'neutral', 3: 'positive'}
df['label'] = df['label'].map(sentiment_mapping)

# Preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters and numbers
    tokens = text.lower().split()
    tokens = [ps.stem(token) for token in tokens if token not in stop_words] # Stemming and remove stop words
    return ' '.join(tokens)

def identify_aspect(text):
    aspect = 'other'
    if 'vaccine' in text:
        aspect = 'vaccine'
    elif 'mask' in text:
        aspect = 'mask'
    elif 'lockdown' in text:
        aspect = 'lockdown'
    return aspect

df['processed_text'] = df['tweet_text'].apply(preprocess_text)
df['aspect'] = df['tweet_text'].apply(identify_aspect)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['processed_text'], df['aspect'], test_size=0.2, random_state=42
)

# Convert processed text data to numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
train_features = tfidf_vectorizer.fit_transform(train_data)
test_features = tfidf_vectorizer.transform(test_data)

# Initialize SVM model
svm_model = SVC(kernel='linear')

# Train the SVM model
svm_model.fit(train_features, train_labels)

# Make predictions on the test set
predictions = svm_model.predict(test_features)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Data Visualization
plt.figure(figsize=(8, 6))
sns.countplot(x=predictions, hue=test_labels, palette='Set2')
plt.title('Aspect-Based Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.legend(title='Aspect')
plt.show()