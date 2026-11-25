"""Feature exploration on the EARLY 80% of each participant's days only.

The last 20% of each participant's timeline is locked away for one final test
(see evaluate_holdout.py) and is never read here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_model as T  # noqa: E402
from features import FEATURE_COLS  # noqa: E402
from extra_features import EXTRA_COLS, add_extra_features, split_by_time  # noqa: E402


def rf(seed=0):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", RandomForestClassifier(400, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1))])


def main():
    raw = T.load_raw_pmdata_if_available()
    df = T.add_next_day_target(add_extra_features(T.engineer_features(raw), T.RAW_DIR))
    early, _locked = split_by_time(df, 0.8)
    y, g = early[T.TARGET], early["participant_id"]
    print(f"exploration rows: {len(early)} (locked rows not used: {len(_locked)})")
    print("persistence  macroF1 %.3f  acc %.3f" % (f1_score(y, early["today_state"], average="macro"), accuracy_score(y, early["today_state"])))

    sets = {"current": FEATURE_COLS, "current+extra": FEATURE_COLS + EXTRA_COLS}
    for name, cols in sets.items():
        p = cross_val_predict(rf(), early[cols], y, groups=g, cv=LeaveOneGroupOut())
        print(f"LOPO 3-class {name:14s} macroF1 %.3f  acc %.3f" % (f1_score(y, p, average="macro"), accuracy_score(y, p)))

    # Personalised setting: within each person, train on their first 75% of the early block, test on the rest.
    tr, te = split_by_time(early, 0.75)
    print("temporal persistence macroF1 %.3f acc %.3f" % (f1_score(te[T.TARGET], te["today_state"], average="macro"), accuracy_score(te[T.TARGET], te["today_state"])))
    for name, cols in sets.items():
        m = rf().fit(tr[cols], tr[T.TARGET])
        p = m.predict(te[cols])
        print(f"temporal 3-class {name:14s} macroF1 %.3f  acc %.3f" % (f1_score(te[T.TARGET], p, average="macro"), accuracy_score(te[T.TARGET], p)))

    # Binary: will tomorrow need a deload?
    yb = (y == "Deload Needed").astype(int)
    pers_b = (early["today_state"] == "Deload Needed").astype(int)
    print("binary persistence  acc %.3f  F1 %.3f" % (accuracy_score(yb, pers_b), f1_score(yb, pers_b)))
    for name, cols in sets.items():
        proba = cross_val_predict(rf(), early[cols], yb, groups=g, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
        pb = (proba >= 0.5).astype(int)
        print(f"LOPO binary {name:14s} AUC %.3f acc %.3f F1 %.3f" % (roc_auc_score(yb, proba), accuracy_score(yb, pb), f1_score(yb, pb)))


if __name__ == "__main__":
    main()
