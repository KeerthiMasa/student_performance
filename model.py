import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Dataset
students = {
    'PreviousScore1': [45, 55, 60, 30, 80, 70, 65, 40, 90, 50],
    'PreviousScore2': [50, 60, 58, 35, 85, 75, 67, 43, 92, 53],
    'Passed': [0, 1, 1, 0, 1, 1, 1, 0, 1, 0]
}

data = pd.DataFrame(students)
X = data[['PreviousScore1', 'PreviousScore2']]
y = data['Passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Optional: print accuracy on test set
predictions = model.predict(X_test)
predicted_labels = [1 if p >= 0.6 else 0 for p in predictions]
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Test accuracy: {accuracy:.2f}")
