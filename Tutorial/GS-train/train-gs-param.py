import os
import re
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


def load_param_dataset(root_dir):
    X = []
    y = []

    pattern = re.compile(
        r"Du([0-9.]+)_Dv([0-9.]+)_F([0-9.]+)_k([0-9.]+)"
    )

    for label_name, label in [("bad", 0), ("good", 1)]:
        folder = os.path.join(root_dir, label_name)

        for fname in os.listdir(folder):
            if not fname.endswith(".png"):
                continue

            match = pattern.search(fname)
            if not match:
                print(f"Skipping {fname}")
                continue

            Du, Dv, F, k = map(float, match.groups())

            X.append([Du, Dv, F, k])
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def make_param_model():
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )

def train_classifier(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.array([1 if label == "good" or label == 1 else 0 for label in y])

    if len(set(y)) < 2:
        raise ValueError(
            "Only one label class was found in this batch. "
            "Increase --batch or sample another batch."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    print("Train an XGBClassifier on", len(X_train), "samples and", len(X_test), "testing samples")

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise ValueError(
            "Only one label class was found in the testing or training. "
            "Increase --batch or sample another batch."
        )

    param_model = make_param_model()
    param_model.fit(X_train, y_train)

    pred = param_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, pred))

    return param_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder with all pre-labeled images")
    parser.add_argument("--out-model", default="param_initial.json", help="Output pre-trained model")
    args = parser.parse_args()
    
    X, y = load_param_dataset(args.folder)

    model = train_classifier(X, y)
        
    model.save_model(args.out_model)
    print("Saved model to", args.out_model)
