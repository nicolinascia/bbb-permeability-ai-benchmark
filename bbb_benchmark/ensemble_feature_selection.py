import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.impute import SimpleImputer, KNNImputer


def ensemble_feature_selection(df, y_class, nan_imp, typesel):
    from sklearn_relief import Relief

    n_features = df.shape[1]
    print("Start num features:", n_features)

    scaler = preprocessing.MinMaxScaler()
    if nan_imp == "zero":
        df0 = df.fillna(0)
    elif nan_imp == "median":
        imputer = SimpleImputer(strategy="median")
        df0 = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif nan_imp == "knn":
        imputer = KNNImputer()
        df0 = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    data = scaler.fit_transform(df0)

    chiSquare = chi2(data, y_class)[0]
    rank_chi2 = np.flip(np.argsort(chiSquare))

    info_gain = mutual_info_classif(data, y_class, random_state=18)
    rank_infogain = np.flip(np.argsort(info_gain))

    forest = RandomForestClassifier(n_jobs=-1, random_state=18)
    forest.fit(data, y_class)
    RF_imp = forest.feature_importances_
    rank_RFimp = np.flip(np.argsort(RF_imp))

    r = Relief(n_features=n_features)
    r.fit_transform(data, y_class)
    rank_relief = np.flip(np.argsort(r.w_))

    rank_matrix = np.column_stack((rank_chi2, rank_infogain, rank_RFimp, rank_relief))
    min_ensemble_features = np.argsort(rank_matrix, axis=0).min(axis=1)
    sort_mfe = np.argsort(min_ensemble_features)
    feat_selected = sort_mfe[0:int(np.floor(np.log2(n_features)))]

    if typesel == "allfeatures":
        df = df.iloc[:, sort_mfe]
    elif typesel == "log2":
        df = df.iloc[:, feat_selected]

    return df
