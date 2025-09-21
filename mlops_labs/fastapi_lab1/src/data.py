import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load the Wine Quality dataset from a CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        X (pandas.DataFrame): The features of the dataset.
        y (pandas.Series): The target values of the dataset.
    """
    data = pd.read_csv(filepath)
    
    # Assuming 'quality' is the target column
    X = data.drop(columns=['quality'])
    y = data['quality']
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (pandas.DataFrame): The features of the dataset.
        y (pandas.Series): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )
    return X_train, X_test, y_train, y_test

# Example usage:
if __name__ == "__main__":
    filepath = r"C:\Users\locha\Downloads\wine_quality_merged.csv"
    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)
