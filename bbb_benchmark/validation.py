from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from .benchmark import fit_selected_model


def run_validation(
    data_path,
    output_dir="results",
    validation_path=None,
    n_jobs=-2,
    resume=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "validation_results.csv"
    pred_path = output_dir / "validation_predictions.csv"

    if resume and result_path.exists() and pred_path.exists():
        return pd.read_csv(result_path), pd.read_csv(pred_path)

    data_path = Path(data_path)
    if validation_path is None:
        validation_path = data_path.parent / "alldesFP_test_data.xlsx"

    test = pd.read_excel(validation_path)

    search, imputer, acf, selected = fit_selected_model(
        data_path,
        representation="alldesFP",
        corr=0.8,
        selection="Ens-log2",
        model_name="RF",
        n_jobs=n_jobs,
    )

    X_test_acf = pd.DataFrame(imputer.transform(test[acf]), columns=acf)
    X_test = X_test_acf[selected]
    y_true = test["Class"].astype(int).reset_index(drop=True)
    y_pred = search.predict(X_test)

    result = pd.DataFrame([{
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }])
    predictions = test[["orig_index", "Class"]].copy()
    predictions["Prediction"] = y_pred

    result.to_csv(result_path, index=False)
    predictions.to_csv(pred_path, index=False)
    return result, predictions
