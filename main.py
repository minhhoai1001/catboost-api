from catboost import CatBoostClassifier
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from json import JSONEncoder

class Item(BaseModel):
    age: List[int]
    sex: List[int]
    cp: List[int]
    trtbps: List[int]
    chol: List[int]
    fbs: List[int]
    restecg: List[int]
    thalachh: List[int]
    exng: List[int]
    oldpeak: List[float]
    slp: List[int]
    caa: List[int]
    thall: List[int]

app = FastAPI()
model = CatBoostClassifier()
model.load_model("model")

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def predict(df):
    scaler = StandardScaler()
    x_predict = scaler.fit_transform(df)
    print(x_predict.T)
    result = model.predict(x_predict.T)
    return result

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict_heart_attack(item: Item):
    # print(item.dict())
    df = pd.DataFrame.from_dict(item.dict(), orient='index')
    # print(df)
    result = predict(df).T
    # label = ["Normal", "High rick heart attack"]
    for pre in result[0]:
        print("Predict: ", pre)
    numpyData = {"Predict": result[0]}
    data = json.dumps(numpyData, cls=NumpyArrayEncoder)
    # print(data)
    return data