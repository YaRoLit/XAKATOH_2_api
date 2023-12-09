import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import Response, FileResponse
from typing import Any
from pydantic import BaseModel
import uvicorn

from request_checker import check_n_fill, fill_nans


class Item(BaseModel):
    BirthDate:          Any # 0   BirthDate             datetime64[ns]
    education:          Any # 1   education             object
    employment_status:  Any # 2   employment status     object
    Value:              Any # 3   Value                 object
    JobStartDate:       Any # 4   JobStartDate          datetime64[ns]
    Position:           Any # 5   Position              object
    MonthProfit:        Any # 6   MonthProfit           float64
    MonthExpense:       Any # 7   MonthExpense          float64
    Gender:             Any # 8   Gender                float64
    Family_status:      Any # 9   Family status         object
    ChildCount:         Any # 10  ChildCount            float64
    SNILS:              Any # 11  SNILS                 float64
    Merch_code:         Any # 17  Merch_code            float64
    Loan_amount:        Any # 18  Loan_amount           float64
    Loan_term:          Any # 19  Loan_term             float64
    Goods_category:     Any # 20  Goods_category        object

app = FastAPI()


@app.get("/")
def root():
    '''Get-запрос к корневому каталогу. Возвращает лого команды.'''
    with open("./images/logo.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.get("/help/")
def find():
    '''Get-запрос для получения описания работы приложения'''
    with open("./images/info.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.get("/info/")
def find():
    '''Get-запрос для получения описания работы приложения'''
    with open("./images/info.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.post("/AskOraqul/")
def get_model_prediction(item: Item):
    '''Отправляем предсказание модели по запрошенной строке данных'''
    # Проверяем полученный json и преобразовываем его в pd.DataFrame
    df = check_n_fill(item)
    # Если в нем содержится хоть один пропуск, заполняем их
    if df.isna().sum().sum():
        df = fill_nans(df)
    # Создаем новые признаки, необходимые для работы модели
    # df = create_feature(df)
    # Загружаем модель
    # model = pickle.load(open('../models/model.pkl', 'rb'))

    return {
        "BankA_decision": 0.82,
        "BankB_decision": 0.56,
        "BankC_decision": 0.01,
        "BankD_decision": 0.23,
        "BankE_decision": 0.00
        }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")