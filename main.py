# main.py
from fastapi import FastAPI
import pandas as pd
from ML.src.preprocessing import gold_transformer
from ML.src.model import train_lgbm, evaluate_lgbm
from ML.src.visualization import plot_predictions
import warnings

app = FastAPI()
warnings.filterwarnings("ignore")

@app.get("/train")
def train_model():
    df = pd.read_csv("dados_normalizados.csv", sep=";") 
    X_train, y_train, X_test, y_test = gold_transformer(df)
    train_lgbm(X_train, y_train)
    results, preds, trues = evaluate_lgbm("models", X_test, y_test)
    plot_predictions(trues.flatten(), preds.flatten())
    return {"message": "Treinamento concluído!", "results": results}
