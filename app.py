from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle

app = FastAPI()

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('account_encoder.pkl', 'rb') as f:
    account_encoder = pickle.load(f)
with open('product_encoder.pkl', 'rb') as f:
    product_encoder = pickle.load(f)

@app.post("/predict")
async def predict_quantity(account_name: str, product_id: str, month: int, year: int):
    try:
        if account_name not in account_encoder.classes_:
            raise HTTPException(status_code=400, detail=f"Account '{account_name}' not found.")
        if product_id not in product_encoder.classes_:
            raise HTTPException(status_code=400, detail=f"Product '{product_id}' not found.")
        if not (1 <= month <= 12):
            raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")
        if year < 2000 or year > 2100:
            raise HTTPException(status_code=400, detail="Year must be reasonable (2000-2100).")

        account_encoded = account_encoder.transform([account_name])[0]
        product_encoded = product_encoder.transform([product_id])[0]

        input_features = pd.DataFrame([[account_encoded, product_encoded, month, year]],
                                      columns=['AccountCode', 'ProductCode', 'MONTH', 'YEAR'])

        predicted_quantity = model.predict(input_features)[0]
        return {"predicted_quantity": round(predicted_quantity)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
















'''from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
import pandas as pd
import pickle
import os
import uvicorn

app = FastAPI()

#API_KEY = os.getenv("API_KEY","salesforecast2025")
#api_key_header = APIKeyHeader(name="X-API-Key")


# Load the model and encoders
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('account_encoder.pkl', 'rb') as f:
    account_encoder = pickle.load(f)
with open('product_encoder.pkl', 'rb') as f:
    product_encoder = pickle.load(f)

@app.post("/predict")
async def predict_quantity(account_name: str, product_id: str, month: int, year: int): #api_key: str= Depends(api_key_header)
    
    #if api_key != API_KEY:
       # raise HTTPException(status_code=401, detail="Invalid API Key")
    
    try:
        # Validate inputs
        if account_name not in account_encoder.classes_:
            raise HTTPException(status_code=400, detail=f"Account '{account_name}' not found.")
        if product_id not in product_encoder.classes_:
            raise HTTPException(status_code=400, detail=f"Product '{product_id}' not found.")
        if not (1 <= month <= 12):
            raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")
        if year < 2000 or year > 2100:
            raise HTTPException(status_code=400, detail="Year must be reasonable (2000-2100).")

        # Encode inputs
        account_encoded = account_encoder.transform([account_name])[0]
        product_encoded = product_encoder.transform([product_id])[0]

        # Create input DataFrame
        input_features = pd.DataFrame([[account_encoded, product_encoded, month, year]],
                                     columns=['AccountCode', 'ProductCode', 'MONTH', 'YEAR'])

        # Predict
        predicted_quantity = model.predict(input_features)[0]
        return {"predicted_quantity": round(predicted_quantity)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")   
async def root():
    return{"message":"Sales Prediction API is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)'''
    
    