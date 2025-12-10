import os
import numpy as np

DATA_DIR = "./data"
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset_final.npz")

def build_dataset():
    X_list = []
    y_list = []
    labels = []

    # Todas as pastas de letras
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

    print("Classes encontradas:", classes)

    for label_index, classe in enumerate(classes):
        classe_dir = os.path.join(DATA_DIR, classe)
        labels.append(classe)

        # pega todos os .npy
        files = [f for f in os.listdir(classe_dir) if f.endswith(".npy")]
        print(f" - Carregando {len(files)} amostras da letra {classe}")

        for f in files:
            path = os.path.join(classe_dir, f)
            sample = np.load(path, allow_pickle=True)
            X_list.append(sample)
            y_list.append(label_index)

    # transforma listas em arrays
    X = np.array(X_list)
    y = np.array(y_list)
    labels = np.array(labels)

    print("Formato final do dataset:")
    print("X:", X.shape)
    print("y:", y.shape)

    # salva compacto
    np.savez(OUTPUT_FILE, X=X, y=y, labels=labels)
    print(f"\nDataset salvo em: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_dataset()
