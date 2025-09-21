import os
import joblib

def predict_data(X, model_path="../model/iris_model.pkl"):
    """
    Predict class labels for the input data using a pre-trained model.

    Args:
        X (pandas.DataFrame or numpy.ndarray): Input features for prediction.
        model_path (str, optional): Path to the saved model file. Defaults to '../model/iris_model.pkl'.

    Returns:
        numpy.ndarray: Predicted class labels.

    Raises:
        FileNotFoundError: If the model file does not exist at the specified path.
        ValueError: If input data X is empty.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    if X is None or len(X) == 0:
        raise ValueError("Input data X is empty.")
    
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    return y_pred
