"""One-shot final test on the LOCKED last 20% of each participant's timeline.

Decisions fixed before this script was first run (from explore_features.py on the
early 80% only): the feature set is features.FEATURE_COLS (the extra load and
personal-baseline features did not help and are not used); model candidates and
inner tuning are train_model.make_candidates() with grouped CV.

Setting: the app has seen each person's earlier days. Train on every participant's
first 80% (by date), test on every participant's last 20%.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_model as T  # noqa: E402
from features import FEATURE_COLS  # noqa: E402
from extra_features import split_by_time  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "models" / "holdout_metrics.json"

# Chosen on the early 80% in explore_more.py, fixed before this script's second run.
# The locked set was first used for the 3-class and plain binary results; the confident-alert
# numbers below are its second use.
CONFIDENT_AT = 0.65


def main():
    df = T.add_next_day_target(T.engineer_features(T.load_raw_pmdata_if_available()))
    train, test = split_by_time(df, 0.8)
    Xtr, ytr, gtr = train[FEATURE_COLS], train[T.TARGET], train["participant_id"].to_numpy()
    Xte, yte = test[FEATURE_COLS], test[T.TARGET]

    # Pick the model family by inner grouped CV on the training block only.
    best_name, best_search = None, None
    for name, (pipe, grid) in T.make_candidates().items():
        search = T.tune(pipe, grid, Xtr, ytr, gtr)
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search
    model = best_search.best_estimator_
    pred = model.predict(Xte)

    pers = test["today_state"]
    res = {
        "setting": "Known users: train on each participant's first 80% of days, test once on their last 20%",
        "train_rows": int(len(train)), "test_rows": int(len(test)), "participants": int(test["participant_id"].nunique()),
        "selected_model": best_name, "selected_params": best_search.best_params_,
        "three_class": {
            "model": {"accuracy": round(accuracy_score(yte, pred), 4), "macro_f1": round(f1_score(yte, pred, average="macro"), 4)},
            "persistence": {"accuracy": round(accuracy_score(yte, pers), 4), "macro_f1": round(f1_score(yte, pers, average="macro"), 4)},
            "per_class": {c: {"precision": round(float(p), 3), "recall": round(float(r), 3)}
                          for c, p, r in zip(T.CLASSES, *precision_recall_fscore_support(yte, pred, labels=T.CLASSES, zero_division=0)[:2])},
        },
    }

    # Binary: will tomorrow need a deload? Same split, same candidates, tuned on the training block.
    ytr_b = (ytr == "Deload Needed").astype(int)
    yte_b = (yte == "Deload Needed").astype(int)
    pipe, grid = T.make_candidates()[best_name]
    mb = T.tune(pipe, grid, Xtr, ytr_b, gtr).best_estimator_
    proba = mb.predict_proba(Xte)[:, list(mb.classes_).index(1)]
    pb = (proba >= 0.5).astype(int)
    pers_b = (pers == "Deload Needed").astype(int)
    res["binary_deload"] = {
        "model": {"accuracy": round(accuracy_score(yte_b, pb), 4), "f1": round(f1_score(yte_b, pb), 4), "auc": round(roc_auc_score(yte_b, proba), 4)},
        "persistence": {"accuracy": round(accuracy_score(yte_b, pers_b), 4), "f1": round(f1_score(yte_b, pers_b), 4)},
        "base_rate": round(float(yte_b.mean()), 4),
    }
    sure = (proba >= CONFIDENT_AT) | (proba <= 1 - CONFIDENT_AT)
    res["binary_deload"]["confident_alerts"] = {
        "threshold": CONFIDENT_AT,
        "coverage": round(float(sure.mean()), 4),
        "days": int(sure.sum()),
        "model_accuracy": round(accuracy_score(yte_b[sure], pb[sure]), 4),
        "persistence_accuracy_same_days": round(accuracy_score(yte_b[sure], pers_b[sure]), 4),
        "model_accuracy_other_days": round(accuracy_score(yte_b[~sure], pb[~sure]), 4),
        "note": "Second use of the locked set; threshold chosen on the early 80% only.",
    }

    # Bootstrap the 3-class macro-F1 difference (model - persistence) over test rows.
    rng = np.random.default_rng(42)
    yv, pv, sv = yte.to_numpy(), np.asarray(pred), pers.to_numpy()
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(yv), len(yv))
        diffs.append(f1_score(yv[i], pv[i], average="macro") - f1_score(yv[i], sv[i], average="macro"))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res["three_class"]["macro_f1_diff_95ci"] = [round(float(lo), 4), round(float(hi), 4)]

    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
