import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "./data/dataset_final.npz"
MODEL_PATH = "./models/model_libras.h5"
OUTPUT_DIR = "./metrics"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def carregar_dataset():
    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    labels = data["labels"]
    return X, y, labels

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.title("Matriz de Confusão")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.close()

def main():
    X, y, labels = carregar_dataset()

    # Dividir dataset: 70% treino, 15% validação, 15% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.18, random_state=42, stratify=y_train
    )

    model = tf.keras.models.load_model(MODEL_PATH)

    # Previsões
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Métricas principais
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels)
    cm = confusion_matrix(y_test, y_pred)

    # Salvar relatório
    with open(f"{OUTPUT_DIR}/report.txt", "w", encoding="utf-8") as f:
        f.write(f"ACCURACY: {accuracy:.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report)

    # Salvar matriz de confusão como imagem
    plot_confusion_matrix(cm, labels)

    print("\n==============================")
    print(" MÉTRICAS GERADAS COM SUCESSO ")
    print("==============================")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório salvo em: metrics/report.txt")
    print("Matriz de confusão salva em: metrics/confusion_matrix.png")

if __name__ == "__main__":
    main()
