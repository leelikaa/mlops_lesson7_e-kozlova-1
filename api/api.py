import uvicorn
import numpy as np
import joblib
import requests

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from data_and_models.data import sc

app = FastAPI()
model_loaded = False
is_alive = True

file_path_to_model_LC = '../models/model_LC.pkl'
file_path_to_model_RF = '../models/model_RF.pkl'
model_path = file_path_to_model_LC

model = joblib.load(file_path_to_model_LC)


def check_model():
    global model_loaded
    try:
        model = joblib.load(model_path)
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        print("Model file not found. Please make sure model.pkl exists.")
    except Exception as e:
        model_loaded = False
        print(f"Error loading model: {str(e)}")


class WineDescription(BaseModel):
    fixed_acidity: float = 7.4
    volatile_acidity: float = 0.7
    citric_acid: float = 0.0
    residual_sugar: float = 1.9
    chlorides: float = 0.076
    free_sulfur_dioxide: float = 11.0
    total_sulfur_dioxide: float = 34.0
    density: float = 0.9978
    pH: float = 3.51
    sulphates: float = 0.56
    alcohol: float = 9.4


class Result(BaseModel):
    result: float


@app.post("/predict", response_model=Result)
def predict(wine_data: WineDescription):
    wine_input = wine_data.dict()
    input_np = np.array(list(wine_input.values())).reshape(1, -1)
    input_scaled = sc.transform(input_np)
    result = model.predict(input_scaled)[0]
    return Result(result=result)


@app.get("/healthcheck")
def healthcheck():
    global model_loaded
    try:
        check_model()
        if model_loaded:
            return {"status": "All good"}
        else:
            raise HTTPException(status_code=503, detail="Model is not found")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Service is unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f" Unexpected Error: {str(e)}")

@app.get("/liveness")
async def liveness():
    if is_alive:
        return {"status": "Ok"}
    raise HTTPException(status_code=503, detail="Service not alive")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
