"""Round 3 of exploration: standard supervised-learning techniques, on the EARLY 80% only.

The locked last 20% of each participant's days is not read here. Every number below is
leave-one-participant-out (LOPO) on the early block: train on 15 people, test on the 16th.

Compared:
  * feature sets: the original features; + lagged self-reports and today's state (features.LAG_COLS);
    + personal normalisation (each value against that person's own running mean and
    spread, using only that day and earlier days);
  * algorithms: logistic regression, k-nearest neighbours, SVM (RBF), naive Bayes,
    decision tree, random forest, extra trees, gradient boosting, histogram gradient
    boosting, a small neural network (MLP), soft voting and stacking ensembles;
  * a structured approach: regress tomorrow's readiness and fatigue, then apply the
    label rule to the predictions.

Writes models/technique_sweep.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_model as T  # noqa: E402
from extra_features import split_by_time  # noqa: E402
from features import BASE_COLS, DERIVED_COLS, LAG_COLS  # noqa: E402

OUT = ROOT / "models" / "technique_sweep.json"

PERSONAL = ["readiness", "fatigue", "soreness", "stress", "sleep_score", "resting_heart_rate", "steps"]
PERSONAL_COLS = [f"{c}_z" for c in PERSONAL]
ORIGINAL_COLS = BASE_COLS + DERIVED_COLS  # the feature set before this round


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Personal normalisation (the lag features come from features.py, shared with serving)."""
    df = df.sort_values(["participant_id", "date"]).copy()
    g = df.groupby("participant_id", sort=False)
    # Each value against this person's own history up to and including today.
    for c in PERSONAL:
        mean = g[c].transform(lambda s: s.expanding(min_periods=3).mean())
        std = g[c].transform(lambda s: s.expanding(min_periods=3).std())
        df[f"{c}_z"] = (df[c] - mean) / std.where(std > 0)
    return df


def imp(*steps):
    return Pipeline([("impute", SimpleImputer(strategy="median")), *steps])


def algorithms():
    lr = imp(("scale", StandardScaler()), ("clf", LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")))
    rf = imp(("clf", RandomForestClassifier(400, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1)))
    hgb = imp(("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=200, l2_regularization=1.0,
                                                    class_weight="balanced", random_state=0)))
    svm = imp(("scale", StandardScaler()), ("clf", SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=0)))
    return {
        "LogisticRegression": lr,
        "KNN (k=25)": imp(("scale", StandardScaler()), ("clf", KNeighborsClassifier(25, weights="distance"))),
        "SVM (RBF)": svm,
        "GaussianNB": imp(("scale", StandardScaler()), ("clf", GaussianNB())),
        "DecisionTree": imp(("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=0))),
        "RandomForest": rf,
        "ExtraTrees": imp(("clf", ExtraTreesClassifier(400, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1))),
        "GradientBoosting": imp(("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=0))),
        "HistGradientBoosting": hgb,
        "MLP": imp(("scale", StandardScaler()), ("clf", MLPClassifier((32,), alpha=1e-2, max_iter=2000, early_stopping=True, random_state=0))),
        "SoftVoting (LR+RF+HGB+SVM)": VotingClassifier([("lr", lr), ("rf", rf), ("hgb", hgb), ("svm", svm)], voting="soft"),
        "Stacking (LR+RF+HGB -> LR)": StackingClassifier([("lr", lr), ("rf", rf), ("hgb", hgb)],
                                                         final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced"), cv=5),
    }


def label_rule(readiness: np.ndarray, fatigue: np.ndarray) -> np.ndarray:
    return np.where((readiness >= 7) & (fatigue >= 3), "Ready",
                    np.where((readiness <= 4) | (fatigue <= 2), "Deload Needed", "Moderate"))


def lopo(model, X, y, groups) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = clone(model).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    return pred


def lopo_regress_then_rule(X, next_r, next_f, groups) -> np.ndarray:
    pred = np.empty(len(X), dtype=object)
    reg = imp(("reg", ExtraTreesRegressor(400, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=-1)))
    for tr, te in LeaveOneGroupOut().split(X, next_r, groups):
        r = clone(reg).fit(X.iloc[tr], next_r.iloc[tr]).predict(X.iloc[te])
        f = clone(reg).fit(X.iloc[tr], next_f.iloc[tr]).predict(X.iloc[te])
        pred[te] = label_rule(np.round(r), np.round(f))
    return pred


def score(y, pred) -> dict:
    return {"accuracy": round(float(accuracy_score(y, pred)), 4), "macro_f1": round(float(f1_score(y, pred, average="macro")), 4)}


def main():
    df = T.add_next_day_target(add_features(T.engineer_features(T.load_raw_pmdata_if_available())))
    # Tomorrow's raw readiness and fatigue, for the regress-then-rule approach.
    full = add_features(T.engineer_features(T.load_raw_pmdata_if_available()))
    nxt = full.groupby("participant_id")[["readiness", "fatigue"]].shift(-1)
    full = full.assign(next_readiness=nxt["readiness"], next_fatigue=nxt["fatigue"])[["participant_id", "date", "next_readiness", "next_fatigue"]]
    df = df.merge(full, on=["participant_id", "date"], how="left")

    early, _ = split_by_time(df, 0.8)  # the locked 20% is never touched here
    early = early.reset_index(drop=True)
    y, groups = early[T.TARGET], early["participant_id"].to_numpy()
    print(f"Early block: {len(early)} rows, {len(np.unique(groups))} participants")

    res = {"rows": int(len(early)), "setting": "LOPO on the early 80% of each participant's days",
           "persistence": score(y, early["today_state"]), "feature_sets": {}, "algorithms": {}, "regress_then_rule": {}}
    print("persistence", res["persistence"])

    sets = {"original": ORIGINAL_COLS, "original+lags": ORIGINAL_COLS + LAG_COLS,
            "original+lags+personal": ORIGINAL_COLS + LAG_COLS + PERSONAL_COLS}
    algos = algorithms()
    # 1) feature sets, with two contrasting algorithms
    for sname, cols in sets.items():
        for aname in ("LogisticRegression", "RandomForest", "HistGradientBoosting"):
            t = time.time()
            s = score(y, lopo(algos[aname], early[cols], y, groups))
            res["feature_sets"].setdefault(sname, {})[aname] = s
            print(f"{sname:24s} {aname:22s} {s}  ({time.time() - t:.0f}s)")
    best_set = max(sets, key=lambda k: np.mean([v["macro_f1"] for v in res["feature_sets"][k].values()]))
    res["chosen_feature_set"] = best_set
    print("chosen feature set:", best_set)

    # 2) every algorithm on that feature set
    cols = sets[best_set]
    for aname, model in algos.items():
        t = time.time()
        s = score(y, lopo(model, early[cols], y, groups))
        res["algorithms"][aname] = s
        print(f"{aname:28s} {s}  ({time.time() - t:.0f}s)")

    # 3) regress tomorrow's readiness and fatigue, then apply the label rule
    ok = early["next_readiness"].notna() & early["next_fatigue"].notna()
    sub = early[ok].reset_index(drop=True)
    s = score(sub[T.TARGET], lopo_regress_then_rule(sub[cols], sub["next_readiness"], sub["next_fatigue"], sub["participant_id"].to_numpy()))
    res["regress_then_rule"] = {**s, "rows": int(len(sub))}
    print("regress-then-rule", s)

    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
