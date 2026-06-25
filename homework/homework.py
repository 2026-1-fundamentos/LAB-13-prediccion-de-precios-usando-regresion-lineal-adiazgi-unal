#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con

# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import os
import gzip
import json
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def pregunta_01():

    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)

    train = pd.read_csv("files/input/train_data.csv.zip")
    test = pd.read_csv("files/input/test_data.csv.zip")

    #
    # Preprocesamiento: crear Age, eliminar Year y Car_Name.
    # Present_Price es el target; Selling_Price se conserva como predictora.
    #
    for df in [train, test]:
        df["Age"] = 2021 - df["Year"]
        df.drop(columns=["Year", "Car_Name"], inplace=True)

    y_train = train["Present_Price"]
    x_train = train.drop(columns=["Present_Price"])

    y_test = test["Present_Price"]
    x_test = test.drop(columns=["Present_Price"])

    #
    # Owner se trata como categórica (pocos valores discretos, sin
    # relación lineal natural con el precio), no como numérica continua.
    #
    cat_cols = ["Fuel_Type", "Selling_type", "Transmission", "Owner"]
    num_cols = ["Selling_Price", "Driven_kms", "Age"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", MinMaxScaler(), num_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selectkbest", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )

    param_grid = {"selectkbest__k": list(range(1, 13))}

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )

    model.fit(x_train, y_train)

    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(model, file)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = [
        {
            "type": "metrics",
            "dataset": "train",
            "r2": float(r2_score(y_train, y_train_pred)),
            "mse": float(mean_squared_error(y_train, y_train_pred)),
            "mad": float(mean_absolute_error(y_train, y_train_pred)),
        },
        {
            "type": "metrics",
            "dataset": "test",
            "r2": float(r2_score(y_test, y_test_pred)),
            "mse": float(mean_squared_error(y_test, y_test_pred)),
            "mad": float(mean_absolute_error(y_test, y_test_pred)),
        },
    ]

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for metric in metrics:
            file.write(json.dumps(metric) + "\n")


if __name__ == "__main__":
    pregunta_01()