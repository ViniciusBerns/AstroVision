import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)


# CONFIGURAÇÕES


IMG_SIZE = 224

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

dataset_path = os.path.join(
    BASE_DIR,
    "dataset"
)

# CARREGAMENTO DO DATASET


imagens = []
labels = []

classes = {
    "flood": 0,
    "wildfire": 1
}

print("\nCarregando imagens...\n")

for classe, label in classes.items():

    pasta = os.path.join(
        dataset_path,
        classe
    )

    for arquivo in os.listdir(pasta):

        caminho = os.path.join(
            pasta,
            arquivo
        )

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

print("Dataset carregado com sucesso!")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

# DIVISÃO TREINO / VALIDAÇÃO / TESTE


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

print("\nTreino:", X_train.shape)
print("Validação:", X_val.shape)
print("Teste:", X_test.shape)


# DATA AUGMENTATION


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])


# ARQUITETURA CNN


model = Sequential([

    Input(shape=(224, 224, 3)),

    data_augmentation,

    Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        1,
        activation="sigmoid"
    )
])


# COMPILAÇÃO


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nResumo do Modelo:\n")
model.summary()


# EARLY STOPPING


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# TREINAMENTO


history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    callbacks=[
        early_stop
    ],

    verbose=1
)


# AVALIAÇÃO


loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n====================================")
print("RESULTADO FINAL")
print("====================================")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# PREDIÇÕES

y_pred = model.predict(X_test)

y_pred = (
    y_pred > 0.5
).astype(int)


# CLASSIFICATION REPORT


print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Flood",
            "Wildfire"
        ]
    )
)


# MATRIZ DE CONFUSÃO


cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Flood",
        "Wildfire"
    ],
    yticklabels=[
        "Flood",
        "Wildfire"
    ]
)

plt.title(
    "Matriz de Confusão"
)

plt.xlabel(
    "Classe Prevista"
)

plt.ylabel(
    "Classe Real"
)

plt.tight_layout()

plt.show()


# GRÁFICO ACCURACY


plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Treino"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validação"
)

plt.title(
    "Accuracy por Época"
)

plt.xlabel(
    "Época"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# GRÁFICO LOSS


plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Treino"
)

plt.plot(
    history.history["val_loss"],
    label="Validação"
)

plt.title(
    "Loss por Época"
)

plt.xlabel(
    "Época"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# SALVAR MODELO


os.makedirs(
    os.path.join(
        BASE_DIR,
        "models"
    ),
    exist_ok=True
)

model.save(
    os.path.join(
        BASE_DIR,
        "models",
        "astrovision.keras"
    )
)

print(
    "\nModelo salvo em models/astrovision.keras"
)
