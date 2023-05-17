from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the feature matrix X and target variable y
X = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15]]
y = [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]

# Create a Random Forest classifier object
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the data
rfc.fit(X, y)

# Make predictions on the training data
y_pred = rfc.predict(X)

# Calculate the accuracy of the model
accuracy = accuracy_score(y, y_pred)

print(f"Accuracy: {accuracy}")