import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('combined_dataset.csv')

# Define features (X) and target (y)
X = df.drop('num', axis=1)
y = df['num']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
