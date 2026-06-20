import argparse
import numpy as np
from xgboost import XGBClassifier


def load_params_model(pretrained_path):
    param_model = XGBClassifier()
    param_model.load_model(pretrained_path)
    return param_model

def find_uncertain_predictions_in_range(
    model,
    Du_range,
    Dv_range,
    F_range,
    k_range,
    sampling_frequency,
    max_probability=None,
):
    """
    Search a grid of Gray-Scott parameters and return the parameter sets
    predicted as good by the trained random forest model.

    Du_range, Dv_range, F_range, k_range:
        Tuples of the form (min_value, max_value).

    sampling_frequency:
        Number of samples per parameter dimension.
        Example: sampling_frequency=10 evaluates 10^4 parameter combinations.

    good_label:
        Label used by the model for good images.
        Use 1 if the model was trained with numeric labels 0/1.

    min_good_probability:
        Optional probability threshold. If None, all predictions equal to
        good_label are returned.

    Returns:
        List of dictionaries with Du, Dv, F, k, predicted_label,
        good_probability, and class_probabilities.
    """

    Du_values = np.linspace(Du_range[0], Du_range[1], sampling_frequency)
    Dv_values = np.linspace(Dv_range[0], Dv_range[1], sampling_frequency)
    F_values = np.linspace(F_range[0], F_range[1], sampling_frequency)
    k_values = np.linspace(k_range[0], k_range[1], sampling_frequency)

    candidates = []
    for Du in Du_values:
        for Dv in Dv_values:
            for F in F_values:
                for k in k_values:
                    candidates.append(
                        [round(float(Du), 5),
                         round(float(Dv), 5),
                         round(float(F), 5),
                         round(float(k), 5)])

    X = np.asarray(candidates, dtype=np.float32)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    probs_label = [max(i) for i in probs]

    predictions = []
    for params, pred, prob_row in zip(X, preds, probs_label):

        if max_probability is not None and prob_row > max_probability:
            continue

        predictions.append(
            {
                "Du": params[0],
                "Dv": params[1],
                "F": params[2],
                "k": params[3],
                "predicted_label": "good" if pred==1 else "bad",
                "probability": prob_row.tolist(),
            }
        )

    predictions.sort(
        key=lambda entry: entry["probability"],
        reverse=False,
    )

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to parameter model, e.g. param_model/param_xgb.json")
    parser.add_argument("--Du-min", type=float, default=0.1, help="Gray-Scott Du parameter value")
    parser.add_argument("--Du-max", type=float, default=0.3, help="Gray-Scott Du parameter value")
    parser.add_argument("--Dv-min", type=float, default=0.08, help="Gray-Scott Dv parameter value")
    parser.add_argument("--Dv-max", type=float, default=0.02, help="Gray-Scott Dv parameter value")
    parser.add_argument("--F-min", type=float, default=0.025, help="Gray-Scott F parameter value")
    parser.add_argument("--F-max", type=float, default=0.005, help="Gray-Scott F parameter value")
    parser.add_argument("--k-min", type=float, default=0.09, help="Gray-Scott k parameter value")
    parser.add_argument("--k-max", type=float, default=0.03, help="Gray-Scott k parameter value")
    parser.add_argument("--sampling-frequency", type=int, default=3, help="Number of samples per parameter dimension. Example: sampling_frequency=10 evaluates 10^4 parameter combinations.")
    parser.add_argument("--num", type=int, default=10, help="Number of returnes samples")
    args = parser.parse_args()

    model = load_params_model(args.model)
    pred = find_uncertain_predictions_in_range(
        model,
        (args.Du_min, args.Du_max),
        (args.Dv_min, args.Dv_max),
        (args.F_min, args.F_max),
        (args.k_min, args.k_max),
        args.sampling_frequency,
    )

    for entry in pred[:args.num]:
        print(entry)