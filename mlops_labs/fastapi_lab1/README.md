Wine Quality Prediction API

This project provides a REST API for predicting wine quality using a trained Decision Tree Classifier. The API is built with FastAPI and exposes endpoints for health checks and quality prediction.

Features
Wine quality prediction endpoin

Accepts numeric wine features including acidity, sugar, alcohol, and type (red/white)

Returns predicted wine quality as an integer

Installation

Clone the repository:
cd mlops_labs/fastapi_lab1


Create a virtual environment (optional but recommended):
python -m venv venv


Install dependencies:

pip install -r requirements.txt

Running the API

Start the FastAPI server:

uvicorn src.main:app --reload --host 127.0.0.1 --port 8000


Open your browser to view the interactive API documentation:

http://127.0.0.1:8000/docs

Response:

{"status": "healthy"}


JSON payload:
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.70,
  "citric_acid": 0.00,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11,
  "total_sulfur_dioxide": 43,
  "density": 0.9978,
  "pH": 3.15,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "type": 0
}

Project Structure
fastapi_lab1/
├─ model/              
├─ src/
│  ├─ main.py           
│  ├─ predict.py         
├─ requirements.txt
├─ README.md