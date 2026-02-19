import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import matplotlib.pyplot as plt
import lightgbm as lgb

from pathlib import Path

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
# Training speed / hardware config
# =========================================================
USE_CUDA = True  # set False to force CPU
N_SPLITS = 3     # CV folds (5 is slower; 3 is usually enough for tuning)
N_TRIALS = 20    # Optuna trials (increase once pipeline works)
EARLY_STOPPING_ROUNDS = 50


def _try_fit_xgb(model: xgb.XGBClassifier, X_tr, y_tr, X_val, y_val):
    try:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        return model
    except Exception as e:
        # Common on Windows if xgboost wasn't built with GPU support
        if USE_CUDA:
            print(f"[WARN] XGBoost GPU fit failed, falling back to CPU. Reason: {e}")
        cpu_params = model.get_params()
        cpu_params.pop("tree_method", None)
        cpu_params.pop("predictor", None)
        cpu_params.pop("gpu_id", None)
        cpu_params["tree_method"] = "hist"
        cpu_params["predictor"] = "auto"
        model_cpu = xgb.XGBClassifier(**cpu_params)
        model_cpu.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        return model_cpu


def _try_fit_lgb(model: lgb.LGBMClassifier, X_tr, y_tr, X_val, y_val):
    try:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        return model
    except Exception as e:
        # Common if installed LightGBM doesn't include GPU support
        if USE_CUDA:
            print(f"[WARN] LightGBM GPU fit failed, falling back to CPU. Reason: {e}")
        cpu_params = model.get_params()
        cpu_params.pop("device_type", None)
        cpu_params["device_type"] = "cpu"
        model_cpu = lgb.LGBMClassifier(**cpu_params)
        model_cpu.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        return model_cpu


# =========================================================
# 1️⃣ Load Dataset
# =========================================================
def main():
    df = pd.read_csv("data/creditcard.csv")

    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()

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

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train.loc[:, ["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test.loc[:, ["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

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
            "tree_method": "gpu_hist" if USE_CUDA else "hist",
            "predictor": "gpu_predictor" if USE_CUDA else "auto",
            "gpu_id": 0,

            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # set high and rely on early stopping for speed
            "n_estimators": trial.suggest_int("n_estimators", 500, 4000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),

            # imbalance handling
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 500)
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, valid_idx in skf.split(X_train, y_train):

            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[valid_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[valid_idx]

            model = xgb.XGBClassifier(**param)
            model = _try_fit_xgb(model, X_tr, y_tr, X_val, y_val)

            y_val_pred = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_val_pred)
            auc_scores.append(auc_score)

        return np.mean(auc_scores)


    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("Best Parameters:")
    print(study.best_trial.params)

    # =========================================================
    # 5️⃣b LightGBM + Cross-Validation + Optuna
    # =========================================================
    print("\n================ LIGHTGBM OPTUNA TUNING ================\n")

    def lgb_objective(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            "metric": "auc",
            "device_type": "gpu" if USE_CUDA else "cpu",
            # hyperparameters
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # set high and rely on early stopping for speed
            "n_estimators": trial.suggest_int("n_estimators", 500, 8000),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            # imbalance handling
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 500.0),
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, valid_idx in skf.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[valid_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[valid_idx]

            model = lgb.LGBMClassifier(**params)
            model = _try_fit_lgb(model, X_tr, y_tr, X_val, y_val)

            y_val_proba = model.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, y_val_proba))

        return float(np.mean(auc_scores))

    lgb_study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    lgb_study.optimize(lgb_objective, n_trials=N_TRIALS)

    print("Best LightGBM Parameters:")
    print(lgb_study.best_trial.params)

    # Train final LightGBM model with best parameters
    print("\n================ FINAL LIGHTGBM MODEL ================\n")
    best_lgb_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "metric": "auc",
        "device_type": "gpu" if USE_CUDA else "cpu",
        **lgb_study.best_trial.params,
    }

    lgb_model = lgb.LGBMClassifier(**best_lgb_params)
    # Use a small eval split for early stopping in final fit
    X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    lgb_model = _try_fit_lgb(lgb_model, X_tr_f, y_tr_f, X_val_f, y_val_f)

    y_pred_lgb = lgb_model.predict(X_test)
    y_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_lgb))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_lgb))

    precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, y_proba_lgb)
    print("PR-AUC:", auc(recall_lgb, precision_lgb))


    # =========================================================
    # 6️⃣ Train Final XGBoost Model
    # =========================================================
    print("\n================ FINAL XGBOOST MODEL ================\n")

    best_params = study.best_trial.params

    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "booster": "gbtree",
        "tree_method": "gpu_hist" if USE_CUDA else "hist",
        "predictor": "gpu_predictor" if USE_CUDA else "auto",
        "gpu_id": 0,
        **best_params,
    }

    final_model = xgb.XGBClassifier(**final_params)
    # Use a small eval split for early stopping in final fit
    X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    final_model = _try_fit_xgb(final_model, X_tr_f, y_tr_f, X_val_f, y_val_f)

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
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, models_dir / "xgboost_model.pkl")
    joblib.dump(lgb_model, models_dir / "lightgbm_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")

    print("\nModel and scaler saved in /models folder")

if __name__ == "__main__":
    main()