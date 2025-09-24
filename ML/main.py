import pandas as pd
from src.preprocessing import gold_transformer, plot_predictions
from src.model import train_lgbm
from src.evaluate import evaluate_lgbm
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    df = pd.read_csv("dados_normalizados.csv", sep=";") 
    X_train, y_train, X_test, y_test = gold_transformer(df)

    print("Treinando modelos...")
    train_lgbm(X_train, y_train)

    print("Avaliando...")
    results, preds, trues = evaluate_lgbm("models", X_test, y_test)
    print(results)

    plot_predictions(trues.flatten(), preds.flatten())
