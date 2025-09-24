import pandas as pd
from ML.src.preprocessing import gold_transformer
from ML.src.model import train_lgbm
from ML.src.evaluate import evaluate_lgbm
from ML.src.visualization import plot_predictions
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("Carregando dados...")
    df = pd.read_csv("dados_normalizados.csv", sep=";") 

    print(" Pré-processando...")
    X_train, y_train, X_test, y_test = gold_transformer(df)

    print("Treinando modelo...")
    train_lgbm(X_train, y_train)

    print("Avaliando modelo...")
    results, preds, trues = evaluate_lgbm("models", X_test, y_test)
    print("Resultados:", results)

    print(" Gerando gráfico de predições...")
    plot_predictions(trues.flatten(), preds.flatten())

    print(" Finalizado com sucesso!")
