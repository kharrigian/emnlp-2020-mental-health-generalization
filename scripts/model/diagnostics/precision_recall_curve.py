
###################
### Configuration
###################

## Choose Model Directory
model_dir = "./models/MODEL_PATH_HERE/"

###################
### Imports
###################

## Standard Library
import os
import sys
import json
import gzip
from glob import glob

## External Library
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

###################
### Load Results
###################

## Splits
with open(f"{model_dir}splits.json","r") as the_file:
    splits = json.load(the_file)
labels = {}
for j in [i["dev"] for _, i in splits["train"].items()]:
    labels.update(j)
labels.update(splits["test"]["1"]["test"])

## Held-out Test Predictions
with gzip.open(f"{model_dir}predictions.tar.gz","r") as the_file:
    test_predictions = json.load(the_file)
test_predictions = test_predictions.get(list(test_predictions.keys())[0])
test_predictions = pd.DataFrame(test_predictions).reset_index().melt(["index"],
                                                                     ["train","test"],
                                                                     value_name="y_pred",
                                                                     var_name="group").set_index("index").dropna()
test_predictions["fold"] = 1
test_predictions["y_true"] = test_predictions.index.map(lambda i: int(labels.get(i) != "control"))
test_predictions["training_group"] = "full"
test_predictions = test_predictions.reset_index(drop=True)

## Cross Validation Predictions
cv_predictions = None
if os.path.exists(f"{model_dir}cross_validation/predictions.csv"):
    cv_predictions = pd.read_csv(f"{model_dir}cross_validation/predictions.csv")
if cv_predictions is not None:
	cv_predictions["training_group"] = "cross_validation"

## Concatenate Predictions
preds = [test_predictions]
if cv_predictions is not None:
    preds.append(cv_predictions)
preds = pd.concat(preds, sort=True)

###################
### Visualize Curves
###################

## Get Groups and Training Groups
groups = preds.groupby(["training_group"])["group"].unique().to_dict()
training_groups = [i for i in ["cross_validation","full"] if i in preds.training_group.unique()]

## Generate Plot
fig, ax = plt.subplots(len(training_groups), 2, figsize=(10,5.8), sharex=True, sharey=True)
for t, tg in enumerate(training_groups):
    tg_groups = [i for i in ["train","dev","test"] if i in groups.get(tg)]
    for g, group in enumerate(tg_groups):
        if len(groups) != 1:
            pax = ax[t, g]
        else:
            pax = ax[g]
        g_data = preds.loc[(preds["training_group"]==tg)&(preds["group"]==group)]
        avg_prec, avg_rec, avg_thres = metrics.precision_recall_curve(g_data["y_true"],
                                                                      g_data["y_pred"],
                                                                      pos_label=1)
        unique_thresh = pd.DataFrame(np.vstack([avg_prec[1:],
                                                avg_rec[1:],
                                                avg_thres]).T).applymap(lambda x: "{:.2f}".format(x)).drop_duplicates().index
        f1_thresh = []
        for p, r, thresh in zip(avg_prec[1:][unique_thresh], avg_rec[1:][unique_thresh], avg_thres[unique_thresh]):
            f1_thresh.append([p,
                              r,
                              thresh,
                              metrics.f1_score(g_data["y_true"],
                                               g_data["y_pred"]>thresh,
                                               pos_label=1)])
        max_f1 = pd.DataFrame(f1_thresh,columns=["precision","recall","thresh","f1"]).sort_values("f1").iloc[-1]
        folds = g_data["fold"].unique()
        for fold in folds:
            fold_g_data = g_data.loc[g_data["fold"]==fold]
            fold_prec, fold_rec, fold_thres = metrics.precision_recall_curve(fold_g_data["y_true"],
                                                                             fold_g_data["y_pred"],
                                                                             pos_label=1)
            pax.plot(fold_prec,
                          fold_rec,
                          color="C0",
                          linestyle="--" if len(folds) > 1 else "-",
                          alpha=0.3 if len(folds) > 1 else 0.8,
                          linewidth=2 if len(folds) == 1 else 1)
        if len(folds) > 1:
            pax.plot(avg_prec,
                          avg_rec,
                          color="C0",
                          alpha=0.8,
                          linewidth=2)
        pax.scatter(max_f1.loc["precision"],
                        max_f1.loc["recall"],
                        label = "Max F1 {:.3f}\n(Threshold: {:.3f})".format(
                                max_f1.loc["f1"], max_f1.loc["thresh"]
                        ),
                        color="crimson",
                        s=30,
                        zorder=10)
        pax.plot([max_f1.loc["precision"], max_f1.loc["precision"]],
                 [0, max_f1.loc["recall"]],
                 color="crimson",
                 linestyle="--",
                 alpha=0.8)
        pax.plot([0, max_f1.loc["precision"]],
                 [max_f1.loc["recall"], max_f1.loc["recall"]],
                 color="crimson",
                 linestyle="--",
                 alpha=0.8)
        pax.legend(loc="upper right",
                        frameon=True,
                        framealpha=0.7)
        if t == len(training_groups) - 1:
            pax.set_xlabel("Precision", fontweight="bold")
        if g == 0:
            pax.set_ylabel("Recall", fontweight="bold")
        pax.set_title("{} ({})".format(tg.replace("_"," ").title(), group.title()),
                           loc="left",
                           fontweight="bold",
                           fontstyle="italic")
        pax.set_xlim(0,1)
        pax.set_ylim(0,1)
fig.tight_layout()
fig.savefig(f"{model_dir}precision_recall_curve.png", dpi=300)
plt.close()
