from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics.pairwise import rbf_kernel
from xgboost import XGBClassifier


MODELS = ("LR", "SVM", "SGD", "RF", "XGB", "HGB")
SELECTIONS = ("Ens-RFA", "Ens-log2", "ACF")

CV_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 18
MODEL_N_JOBS = 1
N_SPLITS = 5
REFIT_METRIC = "average_precision"
SCORING = (
    "accuracy", "balanced_accuracy", "f1", "precision",
    "recall", "roc_auc", "average_precision",
)

MODEL_GRIDS = {
    "LR": {"C": (0.01, 0.1, 1)},
    "SVM": {"C": (0.1, 1), "gamma": (0.01, 0.001, 0.0001)},
    "RF": {
        "n_estimators": (200, 400),
        "max_depth": (5, 10, None),
        "min_samples_leaf": (5, 10, 20),
    },
    "XGB": {
        "n_estimators": (100, 200),
        "max_depth": (2, 3),
        "learning_rate": (0.01, 0.02),
        "subsample": (0.5, 0.6),
        "colsample_bytree": (0.5, 0.6),
    },
    "SGD": {"alpha": (0.001, 0.01, 0.1)},
    "HGB": {
        "l2_regularization": (0.1, 1, 10),
        "min_samples_leaf": (10, 20, 50),
        "learning_rate": (0.01, 0.05),
        "max_depth": (3, 5),
    },
}


def _preprocessor(feature_names):
    fp = [c for c in feature_names if c.startswith("MACCS_")]
    desc = [c for c in feature_names if not c.startswith("MACCS_")]
    return ColumnTransformer([
        ("fingerprints", "passthrough", fp),
        ("descriptors", StandardScaler(), desc),
    ])


def build_model(model_name, feature_names, y):
    prep = _preprocessor(feature_names)
    rs = MODEL_RANDOM_STATE
    n_jobs = MODEL_N_JOBS
    positive_weight = float((y == 0).sum() / (y == 1).sum())

    if model_name == "LR":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=n_jobs, random_state=rs)
        grid = {"clf__C": MODEL_GRIDS["LR"]["C"]}
    elif model_name == "SVM":
        clf = SVC(kernel=rbf_kernel, class_weight="balanced", probability=True, random_state=rs)
        grid = {
            "clf__C": MODEL_GRIDS["SVM"]["C"],
            "clf__gamma": MODEL_GRIDS["SVM"]["gamma"],
        }
    elif model_name == "RF":
        clf = RandomForestClassifier(class_weight="balanced", n_jobs=n_jobs, random_state=rs)
        grid = {
            "clf__n_estimators": MODEL_GRIDS["RF"]["n_estimators"],
            "clf__max_depth": MODEL_GRIDS["RF"]["max_depth"],
            "clf__min_samples_leaf": MODEL_GRIDS["RF"]["min_samples_leaf"],
        }
    elif model_name == "XGB":
        clf = XGBClassifier(
            scale_pos_weight=positive_weight, eval_metric="logloss", n_jobs=n_jobs, random_state=rs
        )
        grid = {f"clf__{k}": v for k, v in MODEL_GRIDS["XGB"].items()}
    elif model_name == "SGD":
        clf = SGDClassifier(
            loss="log_loss", class_weight="balanced", random_state=rs, n_jobs=n_jobs, early_stopping=True
        )
        grid = {"clf__alpha": MODEL_GRIDS["SGD"]["alpha"]}
    elif model_name == "HGB":
        clf = HistGradientBoostingClassifier(
            random_state=rs, early_stopping=True, class_weight="balanced"
        )
        grid = {f"clf__{k}": v for k, v in MODEL_GRIDS["HGB"].items()}
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("prep", prep), ("clf", clf)]), grid
