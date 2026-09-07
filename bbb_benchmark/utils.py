from importlib.resources import files
import pandas as pd
from sklearn.impute import KNNImputer


FEATURE_SETS = pd.read_csv(files("bbb_benchmark").joinpath("resources/feature_sets.csv"))


def get_features(representation, corr, selection, model="ALL"):
    if selection in ("ACF", "Ens-log2"):
        model = "ALL"

    x = FEATURE_SETS[
        (FEATURE_SETS["representation"] == representation)
        & (FEATURE_SETS["corr"].round(1) == round(float(corr), 1))
        & (FEATURE_SETS["selection"] == selection)
        & (FEATURE_SETS["model"] == model)
    ]
    return x.sort_values("order")["feature"].tolist()


def get_acf_features(representation, corr):
    return get_features(representation, corr, "ACF")


def get_selected_features(representation, corr, selection, model=None):
    return get_features(representation, corr, selection, model or "ALL")


def load_training_data(path):
    return pd.read_csv(path)


def prepare_acf_matrix(df, representation, corr):
    acf = get_acf_features(representation, corr)
    imputer = KNNImputer()
    X_acf = pd.DataFrame(imputer.fit_transform(df[acf]), columns=acf)
    y = df["Class"].astype(int).reset_index(drop=True)
    return X_acf, y, imputer, acf


def prepare_selected_matrix(df, representation, corr, selected_features):
    X_acf, y, imputer, acf = prepare_acf_matrix(df, representation, corr)
    return X_acf[selected_features], y, imputer, acf
