import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 224

dataset_path = "dataset"

imagens = []
labels = []

classes = {
    "flood": 0,
    "wildfire": 1
}

for classe, label in classes.items():

    pasta = os.path.join(dataset_path, classe)

    for arquivo in os.listdir(pasta):

        caminho = os.path.join(pasta, arquivo)

        img = cv2.imread(caminho)

        if img is None:
            continue

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        img = img / 255.0

        imagens.append(img)

        labels.append(label)

X = np.array(imagens)

y = np.array(labels)

print("Dataset carregado")

print("X:", X.shape)

print("y:", y.shape)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTreino:")
print(X_train.shape)

print("\nValidação:")
print(X_val.shape)

print("\nTeste:")
print(X_test.shape)