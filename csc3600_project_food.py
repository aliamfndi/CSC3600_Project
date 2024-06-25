# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import random

# Load the csv data
food = pd.read_csv("/content/Food.csv")
food

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
    "With nuts",
    "With milk",
    "With coconut",
    "With seafood",
    "With egg",
    "With gluten",
    "Not spicy"
]

# that is allergic (1)
negative_texts = [
    "No nuts",
    "No milk",
    "No coconut",
    "No seafood",
    "No egg",
    "No gluten",
    "Is spicy"
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
print("Multinomial Accuracy: ", accuracy_mul*100)

# ----Decision Tree----
classifier2 = DecisionTreeClassifier()
classifier2.fit(X_train, y_train)

# Predict labels for the testing set
y_pred_2 = classifier2.predict(X_test)

#  ----Evaluate accuracy----
accuracy_dec = accuracy_score(y_test, y_pred_2)
print("Decision Tree Accuracy: ", accuracy_dec*100)

# List of allergic categories
description = [
    "Nuts",
    "Milk",
    "Coconut",
    "Seafood",
    "Egg",
    "Gluten",
    "Spicy"
]

# New text to classify
new_text = "I want food that have no milk and no gluten"

# Function to check presence of categories
def check_categories(text, positive_categories, negative_categories):
    text_lower = text.lower()
    results = {}
    for category in negative_categories:
        if category.lower() in text_lower:
            results[category] = 1  # if found in negative set to 0
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

file = ["Nuts", "Milk", "Coconut", "Seafood", "Egg", "Gluten", "Spicy"]

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

def recommend_food(food_df, preferences, model):
    filtered_food = food_df.copy()  # Make a copy of the original DataFrame

    X_new = vectorizer.transform(food_df['Food'])  # Replace 'Drink' with the actual column name
    predictions = model.predict(X_new)

    # Filter out food that has any feature with a value of 1 in preferences
    for feature, value in preferences.items():
        if value == 1:
            filtered_food = filtered_food[filtered_food[feature] != 1]

    # Filter food to keep only those that have at least one feature with a value of 0
    mask = False
    for feature, value in preferences.items():
        if value == 0:
            mask |= (filtered_food[feature] == 0)
    filtered_food = filtered_food[mask]

    # If there are no potential foods left, return no matching foods
    if filtered_food.empty:
        return "No matching food found."

    return filtered_food["Food"].tolist()

# Example usage using multinomial naive bayes
recommend_food = recommend_food(food, extracted_values, classifier)
print("Text: ", new_text)
print("")
print("All matching food: ")
for food in recommend_food:
    print("-", food)
if recommend_food != "No matching food found.":
    print("Recommended food: ", random.choice(recommend_food))
else:
    print("")
    print(recommend_food)

def recommend_food(food_df, preferences, model):
    filtered_food = food_df.copy()  # Make a copy of the original DataFrame

    X_new = vectorizer.transform(food_df['Food'])  # Replace 'Drink' with the actual column name
    predictions = model.predict(X_new)

    # Filter out food that has any feature with a value of 1 in preferences
    for feature, value in preferences.items():
        if value == 1:
            filtered_food = filtered_food[filtered_food[feature] != 1]

    # Filter food to keep only those that have at least one feature with a value of 0
    mask = False
    for feature, value in preferences.items():
        if value == 0:
            mask |= (filtered_food[feature] == 0)
    filtered_food = filtered_food[mask]

    # If there are no potential foods left, return no matching foods
    if filtered_food.empty:
        return "No matching food found."

    return filtered_food["Food"].tolist()

# Example usage using decision tree
recommend_food = recommend_food(food, extracted_values, classifier2)
print("Text: ", new_text)
print("")
print("All matching food: ")
for food in recommend_food:
    print("-", food)
if recommend_food != "No matching food found.":
    print("Recommended food: ", random.choice(recommend_food))
else:
    print("")
    print(recommend_food)