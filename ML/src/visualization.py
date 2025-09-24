import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8,5))
    plt.plot(y_true, label="Valores Reais", color="blue")
    plt.plot(y_pred, label="Predições", color="red", linestyle="--")
    plt.legend()
    plt.title("Predições vs Reais")
    plt.xlabel("Amostras")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.show()
