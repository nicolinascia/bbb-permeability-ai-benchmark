from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .models import (
    build_model, MODELS, SELECTIONS, N_SPLITS,
    CV_RANDOM_STATE, REFIT_METRIC, SCORING,
)
from .utils import (
    FEATURE_SETS, load_training_data, prepare_acf_matrix,
    prepare_selected_matrix, get_selected_features,
)


def _grid_search(X, y, model_name, selected, n_jobs):
    model, grid = build_model(model_name, selected, y)
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )
    search = GridSearchCV(
        model,
        grid,
        cv=cv,
        scoring=list(SCORING),
        refit=REFIT_METRIC,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return search


def fit_selected_model(
    data_path,
    representation="alldesFP",
    corr=0.8,
    selection="Ens-log2",
    model_name="RF",
    n_jobs=-2,
):
    df = load_training_data(data_path)
    selected = get_selected_features(representation, corr, selection, model_name)
    X, y, imputer, acf = prepare_selected_matrix(
        df, representation, corr, selected
    )
    search = _grid_search(X, y, model_name, selected, n_jobs)
    return search, imputer, acf, selected


def refit_summary(search, representation, corr, selection, model_name, n_features):
    i = search.best_index_
    cv = search.cv_results_
    return pd.DataFrame([{
        "representation": representation,
        "corr": float(corr),
        "selection": selection,
        "model": model_name,
        "n_features": int(n_features),
        "best_cv_average_precision": float(search.best_score_),
        "best_cv_balanced_accuracy": float(cv["mean_test_balanced_accuracy"][i]),
        "best_params": json.dumps(search.best_params_, sort_keys=True),
    }])


def _benchmark_row(search, representation, corr, selection, model_name, n_features):
    i = search.best_index_
    cv = search.cv_results_
    row = {
        "representation": representation,
        "corr": float(corr),
        "selection": selection,
        "model": model_name,
        "n_features": int(n_features),
        "best_params": json.dumps(search.best_params_, sort_keys=True),
    }
    for metric in SCORING:
        row[f"best_cv_{metric}"] = float(cv[f"mean_test_{metric}"][i])
        row[f"std_cv_{metric}"] = float(cv[f"std_test_{metric}"][i])
    return row


def run_benchmark(
    data_path,
    output_dir="results",
    representations=None,
    correlations=None,
    selections=None,
    models=None,
    n_jobs=-2,
    resume=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"

    if representations is None:
        representations = FEATURE_SETS["representation"].drop_duplicates().tolist()
    if correlations is None:
        correlations = sorted(FEATURE_SETS["corr"].unique())
    if selections is None:
        selections = SELECTIONS
    if models is None:
        models = MODELS

    if resume and result_path.exists():
        results = pd.read_csv(result_path)
    else:
        results = pd.DataFrame()

    done = set()
    if not results.empty:
        done = {
            (r.representation, round(float(r.corr), 1), r.selection, r.model)
            for r in results.itertuples()
        }

    df = load_training_data(data_path)

    for representation in representations:
        for corr in correlations:
            X_acf, y, _, _ = prepare_acf_matrix(df, representation, corr)

            for selection in selections:
                for model_name in models:
                    key = (
                        representation,
                        round(float(corr), 1),
                        selection,
                        model_name,
                    )
                    if key in done:
                        continue

                    selected = get_selected_features(
                        representation, corr, selection, model_name
                    )
                    if not selected:
                        continue

                    search = _grid_search(
                        X_acf[selected], y, model_name, selected, n_jobs
                    )
                    row = _benchmark_row(
                        search,
                        representation,
                        corr,
                        selection,
                        model_name,
                        len(selected),
                    )
                    results = pd.concat(
                        [results, pd.DataFrame([row])],
                        ignore_index=True,
                    )
                    results.to_csv(result_path, index=False)
                    done.add(key)

    if results.empty:
        return results

    return results.sort_values(
        ["best_cv_balanced_accuracy", "n_features"],
        ascending=[False, True],
    ).reset_index(drop=True)


def run_model_refit(
    data_path,
    output_dir="results",
    representation="alldesFP",
    corr=0.8,
    selection="Ens-log2",
    model_name="RF",
    n_jobs=-2,
):
    search, imputer, acf, selected = fit_selected_model(
        data_path, representation, corr, selection, model_name, n_jobs
    )
    summary = refit_summary(
        search, representation, corr, selection, model_name, len(selected)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "model_refit.csv", index=False)
    return summary, search, imputer, acf, selected
