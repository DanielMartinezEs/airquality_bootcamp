# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd

app = FastAPI()

class Item(BaseModel):
    NOx(GT): float
    NO2(GT): float
    PT08.S4(NO2): float
    PT08.S5(O3): float
    T: float
    RH: float
    AH: float

@app.post("/predict")
async def predict(features: Item):
    try:
        # Load your pre-trained machine learning model
        model = joblib.load('model.joblib')

        # Prepare input features for prediction
        features_df = pd.DataFrame([features.dict()])

        # Make predictions
        prediction = model.predict(features_df)

        # Return the prediction as JSON response
        return {"CO(GT)": prediction[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)