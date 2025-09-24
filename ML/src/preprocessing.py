import numpy as np
import pandas as pd

def create_sequences_multi_step(X, y, context_len=45, horizon=7):
    X_seq, y_seq = [], []
    for i in range(context_len, len(X) - horizon + 1):
        X_seq.append(X.iloc[i - context_len:i].values)
        y_seq.append(y.iloc[i:i + horizon].values.flatten())
    return np.array(X_seq), np.array(y_seq)

def gold_transformer(data, target="val_volumeutilcon", cat_col="id_reservatorio"):
    train_list, test_list = [], []

    for rid, group in data.groupby(cat_col):
        group = group.sort_values(["ano", "mes", "dia"])
        split_idx = int(len(group) * 0.7)

        train_part = group.iloc[:split_idx]
        test_part = group.iloc[split_idx:]

        # Média incremental como feature
        train_part["id_media"] = train_part[target].expanding().mean().shift(1)
        global_mean = train_part[target].mean()
        train_part["id_media"].fillna(global_mean, inplace=True)

        # Expansão para incluir test
        full_series = pd.concat([train_part, test_part])
        full_series["id_media"] = full_series[target].expanding().mean().shift(1)

        test_part["id_media"] = full_series.loc[test_part.index, "id_media"]
        test_part["id_media"].fillna(global_mean, inplace=True)

        train_list.append(train_part)
        test_list.append(test_part)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    test_df.fillna(0, inplace=True)

    # Features
    features = [
        "id_media",
        "val_volmax",
        "ear_reservatorio_percentual_lag1",
        "ear_reservatorio_percentual_lag7",
        "ear_reservatorio_percentual_roll7",
        "dia",
        "mes",
        "ano"
    ]

    X_train = train_df[features]
    y_train = train_df[[target]]

    X_test = test_df[features]
    y_test = test_df[[target]]

    X_train_seq, y_train_seq = create_sequences_multi_step(X_train, y_train)
    X_test_seq, y_test_seq = create_sequences_multi_step(X_test, y_test)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq
