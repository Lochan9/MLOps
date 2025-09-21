from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data

app = FastAPI(title="Wine Quality Prediction API")

# Input schema for wine features
class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    type: int  # 0 for red, 1 for white

# Output schema
class WineResponse(BaseModel):
    quality: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    """
    Predict the quality of a wine sample.
    """
    try:
        features = [[
            wine_features.fixed_acidity,
            wine_features.volatile_acidity,
            wine_features.citric_acid,
            wine_features.residual_sugar,
            wine_features.chlorides,
            wine_features.free_sulfur_dioxide,
            wine_features.total_sulfur_dioxide,
            wine_features.density,
            wine_features.pH,
            wine_features.sulphates,
            wine_features.alcohol,
            wine_features.type
        ]]

        prediction = predict_data(features, model_path="../model/wine_model.pkl")
        return WineResponse(quality=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
