import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK resources (if not already done)
import nltk
nltk.download('stopwords')

# Load your dataset
# Assuming 'your_dataset.csv' contains 'label' and 'tweet_text' columns
df = pd.read_csv('covid-19_tweets.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters and numbers
    tokens = text.lower().split()
    tokens = [ps.stem(token) for token in tokens if token not in stop_words] # Stemming and remove stop words
    return ' '.join(tokens)

df['processed_text'] = df['tweet_text'].apply(preprocess_text)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['processed_text'], df['label'], test_size=0.2, random_state=42
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