import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib

def train_lgbm(X, y, n_lags=45, horizon=7):
    X = X.reshape(X.shape[0], n_lags * X.shape[2])
    y_df = pd.DataFrame(y, columns=[f"t+{i+1}" for i in range(horizon)])

    models = {}
    for step in range(horizon):
        print(f"Treinando modelo para t+{step+1}")
        y_step = y_df.iloc[:, step]
        train_data = lgb.Dataset(X, label=y_step)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31
        }

        model = lgb.train(params, train_data, num_boost_round=50)
        models[f"t+{step+1}"] = model

        joblib.dump(model, f"lgbm_model_t{step+1}.pkl")

    return models

def lgb_predictor(context_window: np.ndarray, horizon=7, n_lags=45, n_features=8) -> np.ndarray:
    preds = []

    if context_window.shape != (n_lags, n_features):
        raise ValueError(f"Esperado shape {(n_lags, n_features)}, mas recebi {context_window.shape}")

    current_window = context_window.reshape(1, -1)

    for step in range(1, horizon + 1):
        model = joblib.load(f"lgbm_model_t{step}.pkl")
        pred = model.predict(current_window)[0]
        preds.append(float(pred))

        flat = current_window.reshape(n_lags, n_features)
        flat[-1, 0] = pred  # Atualiza a feature alvo

        current_window = flat.reshape(1, -1)

    return np.array(preds)
