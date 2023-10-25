from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from mlops_hse.two_classes import AllModels

app = FastAPI(title="MLApp")
allmodels = AllModels()


class ClassType(Enum):
    cb = "cb"
    rf = "rf"


class FullType(Enum):
    cb = "<class 'catboost.core.CatBoostClassifier'>"
    rf = "<class 'sklearn.ensemble._forest.RandomForestClassifier'>"


class Model(BaseModel):
    user_model_name: str
    type_model: ClassType
    params: Dict
    fitted: bool


class ModelTypes(BaseModel):
    model_name: ClassType
    model_type: FullType


class Data(BaseModel):
    X: Dict[str, List[float]]
    y: Optional[List[float]]


@app.get("/get_available_model_types", status_code=200)
async def getting_available_model_types() -> List[ModelTypes]:
    return allmodels.get_available_model_types(show=True)


@app.get("/get_models", status_code=200)
async def getting_models(
    only_fitted: bool | None = False, name_model: str | None = None
) -> List[Model]:
    return allmodels.get_models(only_fitted, name_model)


@app.post("/init_new_model", status_code=201)
async def init_models(
    type_model: str,
    user_model_name: str,
    params: Dict = {"random_state": 42, "n_estimators": 100},
) -> Model:
    return allmodels.init_new_model(type_model, user_model_name, params=params)


@app.put("/model_fit/{user_model_name}", status_code=200)
async def model_fit(user_model_name: str, data: Data) -> Model:
    list_values = list(data.X.values())
    X = np.zeros((len(list_values[0]), len(data.X)))
    y = np.array(data.y)
    for feature_id in range(len(list(data.X.keys()))):
        X[:, feature_id] = list_values[feature_id]
    allmodels.model_fit(X, y, user_model_name)
    return allmodels.get_models(name_models=user_model_name)[0]


@app.put("/model_predict/{user_model_name}", status_code=200)
async def model_predict(user_model_name: str, data: Data) -> Dict[str, List]:
    list_values = list(data.X.values())
    X = np.zeros((len(list_values[0]), len(data.X)))
    for feature_id in range(len(list(data.X.keys()))):
        X[:, feature_id] = list_values[feature_id]
    return {
        "preds": list(allmodels.model_predict(X, user_model_name).flatten())
    }


@app.delete("/delete_model/{user_model_name}", status_code=200)
async def delete_model(user_model_name: str) -> List[Model]:
    allmodels.delete_model(user_model_name)
    return allmodels.get_models()


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("mlops_hse.main:app", host="127.0.0.1", port=8005, reload=True)
