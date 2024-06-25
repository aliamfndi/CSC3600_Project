
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import random

# Load the csv data
drink = pd.read_csv("/content/Drink.csv")
drink

# Use natural language processing (NLP) technique
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Use NLP to filter the text
positive_texts = [
    "With coffee",
    "With milk",
    "With nut",
    "With matcha",
]

negative_texts = [
    "No coffee",
    "No milk",
    "No nut",
    "No matcha",
]

# Positive and negative text
texts = positive_texts + negative_texts

# Positive text: 1     Negative text: 0
labels = [1] * len(positive_texts) + [0] * len(negative_texts)

processed_texts = []
stop_words = set(stopwords.words('english'))
print("")

for text in texts:
    # Tokenize the review
    words = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Join the words back into a single string
    processed_text = ' '.join(words)
    processed_texts.append(processed_text)

    # Print the output
    print("Original sentence: ", texts)
    print("Sentence in word: ", words)
    sentence_with_category = nltk.pos_tag(words)
    print("Sentence in word with category: ", sentence_with_category)
    print("")

# Count vectorizer to handle text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# ----Multinomial Naive Bayes----
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict labels for the testing set
y_pred = classifier.predict(X_test)

#  ----Evaluate accuracy----
accuracy_mul = accuracy_score(y_test, y_pred)
print("Multinomial Naive Bayes' Accuracy: ", accuracy_mul*100)

# ----Decision Tree----
classifier2 = DecisionTreeClassifier()
classifier2.fit(X_train, y_train)

# Predict labels for the testing set
y_pred_2 = classifier2.predict(X_test)

#  ----Evaluate accuracy----
accuracy_dec = accuracy_score(y_test, y_pred_2)
print("Decision Tree Accuracy: ", accuracy_dec*100)

# List of categories or labels
description = [
    "Hot",
    "Cold",
    "Ice blended",
    "Coffee",
    "Milk",
    "Nut",
    "Matcha"
]

condition = [
    "cold",
    "hot",
    "ice blended"
]

# New text to classify
new_text = "I want hot drink with no matcha" # Modidfy the text here

# Function to check presence of categories
def check_categories(text, positive_categories, negative_categories):
    text_lower = text.lower()
    results = {}
    for category in negative_categories:
        if category.lower() in text_lower:
            results[category] = 0  # if found in negative set to 0
        else:
            results[category] = 1
    for category in condition:
        if category.lower() in text_lower:
            results[category] = 1  # if found in condition set to 1
        else:
            results[category] = 0

    return results

# Generate results
results = check_categories(new_text, positive_texts, negative_texts)

text_results = []

# Iterate through results.items() and format each key-value pair
for category, value in results.items():
    text_results.append(f"{category}: {value}")

# Print the formatted results list
print(text_results)

file = ["Coffee", "Milk", "Nut", "Matcha", "Cold", "Hot", "Ice blended"]

# Initialize an empty dictionary to store extracted values
extracted_values = {}

# Zip together file and formatted_results and iterate through them
for category, result in zip(file, text_results):
    # Split the result string to extract the value and convert to integer
    value = int(result.split(": ")[1])
    # Store the value in the dictionary with the category as key
    extracted_values[category] = value

# Print extracted values dictionary
print("Extracted values:")
print(extracted_values)

"""The one below is using multinomial naive bayes"""

def recommend_drink(drink_df, preferences, model):
    filtered_drinks = drink_df.copy()  # Make a copy of the original DataFrame

    X_new = vectorizer.transform(drink_df['Drink'])  # Replace 'Drink' with the actual column name
    predictions = model.predict(X_new)

    # Filter out drinks that have any feature with a value of 0 in preferences
    for feature, value in preferences.items():
        if value == 0:
            filtered_drinks = filtered_drinks[filtered_drinks[feature] != 1]

    # Filter drinks to keep only those that match at least one positive preference (value of 1)
    mask = False
    for feature, value in preferences.items():
        if value == 1:
            mask |= (filtered_drinks[feature] == 1)
    filtered_drinks = filtered_drinks[mask]

    # If there are no potential drinks left, return no matching drinks
    if filtered_drinks.empty:
        return "No matching drinks found."

    return filtered_drinks["Drink"].tolist()

# Example usage using decision tree
recommended_drinks = recommend_drink(drink, extracted_values, classifier)
print("Text: ", new_text)
print("")
print("All matching drinks: ")
for drink in recommended_drinks:
    print(drink)
if recommended_drinks != "No matching drinks found.":
    print("Recommended drink: ", random.choice(recommended_drinks))
else:
    print(recommended_drinks)

"""The one below is using decision tree"""

def recommend_drink(drink_df, preferences, model):
    filtered_drinks = drink_df.copy()  # Make a copy of the original DataFrame

    X_new = vectorizer.transform(drink_df['Drink'])  # Replace 'Drink' with the actual column name
    predictions = model.predict(X_new)

    # Filter out drinks that have any feature with a value of 0 in preferences
    for feature, value in preferences.items():
        if value == 0:
            filtered_drinks = filtered_drinks[filtered_drinks[feature] != 1]

    # Filter drinks to keep only those that match at least one positive preference (value of 1)
    mask = False
    for feature, value in preferences.items():
        if value == 1:
            mask |= (filtered_drinks[feature] == 1)
    filtered_drinks = filtered_drinks[mask]

    # If there are no potential drinks left, return no matching drinks
    if filtered_drinks.empty:
        return "No matching drinks found."

    return filtered_drinks["Drink"].tolist()

# Example usage using decision tree
recommended_drinks = recommend_drink(drink, extracted_values, classifier2)
print("Text: ", new_text)
print("")
print("All matching drinks: ")
for drink in recommended_drinks:
    print(drink)
if recommended_drinks != "No matching drinks found.":
    print("Recommended drink: ", random.choice(recommended_drinks))
else:
    print(recommended_drinks)