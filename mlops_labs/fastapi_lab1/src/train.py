import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

def load_data(filepath):
    """
    Load and preprocess the Wine Quality dataset from a CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        X (pandas.DataFrame): The features of the dataset (numeric).
        y (pandas.Series): The target values of the dataset.
    """
    data = pd.read_csv(filepath)

    # Handle categorical column 'type' (red/white wine)
    if "type" in data.columns:
        le = LabelEncoder()
        data["type"] = le.fit_transform(data["type"])  # white → 1, red → 0

    X = data.drop(columns=["quality"])
    y = data["quality"]
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    """
    return train_test_split(X, y, test_size=0.3, random_state=12)

def fit_model(X_train, y_train):
    """
    Train a Decision Tree Classifier and save the model.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    joblib.dump(dt_classifier, "../model/wine_model.pkl")
    print("✅ Model trained and saved to ../model/wine_model.pkl")

if __name__ == "__main__":
    filepath = r"C:\Users\locha\Downloads\wine_quality_merged.csv"
    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
