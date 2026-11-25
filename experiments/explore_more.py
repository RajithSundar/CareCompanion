"""Round 2 of exploration, again on the EARLY 80% of each participant's days only.

Ideas: lagged self-reports (t-1, t-2) and their trend; weighting a person's own
history; and selective prediction (accuracy when the model is confident, with coverage).
Inner check: within the early block, train on each person's first 75% and test on the rest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_model as T  # noqa: E402
from features import FEATURE_COLS  # noqa: E402
from extra_features import split_by_time  # noqa: E402

LAG_COLS = ["readiness_lag1", "readiness_lag2", "fatigue_lag1", "fatigue_lag2", "readiness_trend3", "soreness_lag1", "stress_lag1"]


def add_lags(df):
    df = df.sort_values(["participant_id", "date"]).copy()
    g = df.groupby("participant_id")
    for c in ("readiness", "fatigue", "soreness", "stress"):
        df[f"{c}_lag1"] = g[c].shift(1)
    for c in ("readiness", "fatigue"):
        df[f"{c}_lag2"] = g[c].shift(2)
    df["readiness_trend3"] = df["readiness"] - df["readiness_lag2"]
    return df


def rf(**kw):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", RandomForestClassifier(500, min_samples_leaf=5, max_depth=8, class_weight="balanced", random_state=0, n_jobs=-1, **kw))])


def main():
    df = T.add_next_day_target(add_lags(T.engineer_features(T.load_raw_pmdata_if_available())))
    early, _ = split_by_time(df, 0.8)
    tr, te = split_by_time(early, 0.75)
    yt, ye = tr[T.TARGET], te[T.TARGET]
    print(f"inner train {len(tr)}  inner test {len(te)}")
    print("persistence 3-class acc %.3f F1 %.3f" % (accuracy_score(ye, te.today_state), f1_score(ye, te.today_state, average="macro")))

    sets = {"current": FEATURE_COLS, "current+lags": FEATURE_COLS + LAG_COLS}
    for name, cols in sets.items():
        p = rf().fit(tr[cols], yt).predict(te[cols])
        print(f"3-class {name:14s} acc %.3f F1 %.3f" % (accuracy_score(ye, p), f1_score(ye, p, average="macro")))

    # Personal weighting: a person's own rows count 3x when predicting them (one model per person).
    for cols_name, cols in sets.items():
        preds = pd.Series(index=te.index, dtype=object)
        for pid in te.participant_id.unique():
            w = np.where(tr.participant_id == pid, 3.0, 1.0)
            m = rf().fit(tr[cols], yt, c__sample_weight=w)
            idx = te.participant_id == pid
            preds[idx] = m.predict(te.loc[idx, cols])
        print(f"3-class personal x3 {cols_name:10s} acc %.3f F1 %.3f" % (accuracy_score(ye, preds), f1_score(ye, preds, average="macro")))

    # Binary rest-day alert, with selective prediction.
    ytb, yeb = (yt == "Deload Needed").astype(int), (ye == "Deload Needed").astype(int)
    pers = (te.today_state == "Deload Needed").astype(int)
    print("binary persistence acc %.3f F1 %.3f" % (accuracy_score(yeb, pers), f1_score(yeb, pers)))
    for name, cols in sets.items():
        proba = rf().fit(tr[cols], ytb).predict_proba(te[cols])[:, 1]
        pb = (proba >= 0.5).astype(int)
        print(f"binary {name:14s} acc %.3f F1 %.3f AUC %.3f" % (accuracy_score(yeb, pb), f1_score(yeb, pb), roc_auc_score(yeb, proba)))
        for thr in (0.65, 0.7, 0.75, 0.8):
            sure = (proba >= thr) | (proba <= 1 - thr)
            if sure.sum():
                print(f"   confident at {thr:.2f}: covers %.0f%% of days, acc %.3f" % (100 * sure.mean(), accuracy_score(yeb[sure], pb[sure])))


if __name__ == "__main__":
    main()
