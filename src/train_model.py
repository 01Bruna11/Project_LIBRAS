import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import os

DATASET_PATH = "./data/dataset_final.npz"
MODEL_PATH = "./models/model_libras.h5"

def carregar_dataset():
    data = np.load(DATASET_PATH, allow_pickle=True)
    return data["X"], data["y"], data["labels"]

if __name__ == "__main__":
    X, y, labels = carregar_dataset()

    num_classes = len(labels)
    input_dim = X.shape[1] 

    # separa treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelo simples
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Treinando modelo...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

    os.makedirs("../models", exist_ok=True)
    model.save(MODEL_PATH)

    print(f"\nModelo salvo com sucesso em: {MODEL_PATH}")

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nAcurácia no teste: {acc:.4f}")
