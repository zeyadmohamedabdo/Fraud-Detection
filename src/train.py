import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)

# =========================================================
# 1️⃣ Load Dataset
# =========================================================
def main():
    df = pd.read_csv("data/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # =========================================================
    # 2️⃣ Stratified Split
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =========================================================
    # 3️⃣ Scaling (Time & Amount only)
    # =========================================================
    scaler = StandardScaler()

    X_train[["Time", "Amount"]] = scaler.fit_transform(
        X_train[["Time", "Amount"]]
    )

    X_test[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )

    # =========================================================
    # 4️⃣ Baseline Model (Logistic Regression)
    # =========================================================
    print("\n================ BASELINE MODEL ================\n")

    lr = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    )

    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_lr))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))

    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_proba_lr)
    print("PR-AUC:", auc(recall_lr, precision_lr))


    # =========================================================
    # 5️⃣ XGBoost + Optuna
    # =========================================================
    print("\n================ XGBOOST OPTUNA TUNING ================\n")

    def objective(trial):

        param = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
            "booster": "gbtree",

            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),

            # imbalance handling
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 500)
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, valid_idx in skf.split(X_train, y_train):

            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[valid_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[valid_idx]

            model = xgb.XGBClassifier(**param)

            model.fit(X_tr, y_tr, verbose=False)

            y_val_pred = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_val_pred)
            auc_scores.append(auc_score)

        return np.mean(auc_scores)


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best Parameters:")
    print(study.best_trial.params)


    # =========================================================
    # 6️⃣ Train Final XGBoost Model
    # =========================================================
    print("\n================ FINAL XGBOOST MODEL ================\n")

    best_params = study.best_trial.params

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    y_pred_xgb = final_model.predict(X_test)
    y_proba_xgb = final_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_xgb))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))

    precision_xgb, recall_xgb, thresholds = precision_recall_curve(y_test, y_proba_xgb)
    print("PR-AUC:", auc(recall_xgb, precision_xgb))


    # =========================================================
    # 7️⃣ Threshold Tuning
    # =========================================================
    print("\n================ THRESHOLD TUNING ================\n")

    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.05):

        y_pred_thresh = (y_proba_xgb > threshold).astype(int)

        report = classification_report(
            y_test,
            y_pred_thresh,
            output_dict=True
        )

        f1 = report["1"]["f1-score"]

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("Best Threshold:", best_threshold)
    print("Best F1 Score:", best_f1)


    # =========================================================
    # 8️⃣ Save Model
    # =========================================================
    joblib.dump(final_model, "models/xgboost_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nModel and scaler saved in /models folder")

if __name__ == "__main__":
    main()