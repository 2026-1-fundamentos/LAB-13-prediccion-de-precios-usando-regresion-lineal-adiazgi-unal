import pandas as pd
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

train = pd.read_csv("files/input/train_data.csv.zip")
test = pd.read_csv("files/input/test_data.csv.zip")

for df in [train, test]:
    df["Age"] = 2021 - df["Year"]
    df.drop(columns=["Year", "Car_Name"], inplace=True)

y_train = train["Present_Price"]
x_train = train.drop(columns=["Present_Price"])
y_test = test["Present_Price"]
x_test = test.drop(columns=["Present_Price"])

R2_TRAIN_MIN = 0.889
MSE_TRAIN_MAX = 5.950
MAD_TRAIN_MAX = 1.600
R2_TEST_MIN = 0.728
MSE_TEST_MAX = 32.910
MAD_TEST_MAX = 2.430
SCORE_TRAIN_MAX = -1.590
SCORE_TEST_MAX = -2.429


def evaluate(name, cat_cols, num_cols, k_fixed):
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
    param_grid = {"selectkbest__k": [k_fixed]}

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    r2_tr = r2_score(y_train, y_train_pred)
    mse_tr = mean_squared_error(y_train, y_train_pred)
    mad_tr = mean_absolute_error(y_train, y_train_pred)
    r2_te = r2_score(y_test, y_test_pred)
    mse_te = mean_squared_error(y_test, y_test_pred)
    mad_te = mean_absolute_error(y_test, y_test_pred)
    score_tr = model.score(x_train, y_train)
    score_te = model.score(x_test, y_test)

    checks = {
        "r2_tr>min": r2_tr > R2_TRAIN_MIN,
        "mse_tr<max": mse_tr < MSE_TRAIN_MAX,
        "mad_tr<max": mad_tr < MAD_TRAIN_MAX,
        "score_tr<max": score_tr < SCORE_TRAIN_MAX,
        "r2_te>min": r2_te > R2_TEST_MIN,
        "mse_te<max": mse_te < MSE_TEST_MAX,
        "mad_te<max": mad_te < MAD_TEST_MAX,
        "score_te<max": score_te < SCORE_TEST_MAX,
    }
    n_pass = sum(checks.values())
    status = "TODO PASA!!!" if n_pass == 8 else f"{n_pass}/8"

    print(f"{name} | k={k_fixed} | {status}")
    print(f"  train: r2={r2_tr:.4f} mse={mse_tr:.4f} mad={mad_tr:.4f} score={score_tr:.4f}")
    print(f"  test:  r2={r2_te:.4f} mse={mse_te:.4f} mad={mad_te:.4f} score={score_te:.4f}")
    if n_pass < 8:
        failed = [k for k, v in checks.items() if not v]
        print(f"  FALLA: {failed}")
    print()
    return n_pass == 8


configs = {
    "Owner_numerica": (["Fuel_Type", "Selling_type", "Transmission"], ["Selling_Price", "Driven_kms", "Owner", "Age"]),
    "Owner_categorica": (["Fuel_Type", "Selling_type", "Transmission", "Owner"], ["Selling_Price", "Driven_kms", "Age"]),
}

found_any = False
for cfg_name, (cat_cols, num_cols) in configs.items():
    for k in range(1, 15):
        ok = evaluate(f"{cfg_name}", cat_cols, num_cols, k_fixed=k)
        if ok:
            found_any = True

print("=" * 50)
print("¿Se encontró alguna combinación que pase TODO?", found_any)