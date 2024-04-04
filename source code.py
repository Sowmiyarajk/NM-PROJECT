# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (features and labels)
# Assume features are extracted network activity data
# and labels indicate whether the activity is benign (0) or malicious (1)
features = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]]  # Sample features
labels = [0, 1, 0, 1]  # Corresponding labels (0: benign, 1: malicious)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the testing set
predictions = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Example usage: Predict whether a new network activity is benign or malicious
new_activity = [[0, 1, 1]]  # Example new network activity
prediction = classifier.predict(new_activity)
if prediction[0] == 0:
    print("The new activity is benign.")
else:
    print("The new activity is malicious.")