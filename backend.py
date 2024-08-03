import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Try reading the CSV file with a different encoding
try:
    data = pd.read_csv('spam_data.csv', encoding='latin1')
except UnicodeDecodeError:
    data = pd.read_csv('spam_data.csv', encoding='ISO-8859-1')


# Preprocess the data
X = data['v2']
y = data['v1']

# Convert text to numerical data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

    # Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)



