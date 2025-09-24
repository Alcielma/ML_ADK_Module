import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.model import lgb_predictor

def evaluate_lgbm(models_dir, X_test, y_test, horizon=7, n_lags=45, n_features=8):
    all_preds, all_true = [], []

    for seq, y_true in zip(X_test, y_test):
        preds = lgb_predictor(seq, horizon=horizon, n_lags=n_lags, n_features=n_features)
        all_preds.append(preds)
        all_true.append(y_true.flatten())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    results = {}

    for step in range(horizon):
        mse = mean_squared_error(all_true[:, step], all_preds[:, step])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_true[:, step], all_preds[:, step])

        results[f"t+{step+1}"] = {"MSE": mse, "RMSE": rmse, "MAE": mae}

    mse_total = mean_squared_error(all_true.flatten(), all_preds.flatten())
    rmse_total = np.sqrt(mse_total)
    mae_total = mean_absolute_error(all_true.flatten(), all_preds.flatten())
    r2 = r2_score(all_true.flatten(), all_preds.flatten())

    results["Overall"] = {"MSE": mse_total, "RMSE": rmse_total, "MAE": mae_total, "R2": r2}

    return results, all_preds, all_true

def plot_predictions(trues, preds):
    plt.figure(figsize=(7, 7))
    plt.scatter(trues, preds, alpha=0.5, label="Previsões")
    plt.plot([trues.min(), trues.max()],
             [trues.min(), trues.max()],
             'r--', lw=2, label="Ideal (y=x)")

    plt.xlabel("Valores Reais (y_test)")
    plt.ylabel("Valores Previstos (y_pred)")
    plt.title("Dispersão: Real vs Previsto")
    plt.legend()
    plt.grid(True)
    plt.show()

