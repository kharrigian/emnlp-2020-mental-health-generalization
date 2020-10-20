
## Prefix (Unique Identifier for Experiments)
PREFIX = "Downsampled_Balanced"

## Output
PLOT_DIR = "./plots/"
DATA_DIR = "./data/results/domain_transfer/"
CACHE_DIR = "./data/results/cache/"

#######################
### Imports
#######################

## Standard Library
import os
import json
from glob import glob

## External Libaries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics

#######################
### Functions
#######################

def load_config(config_file):
    """
    Load configuration file for an experiment

    Args:
        config_file (str): Path to the configuration file
    
    Returns:
        config (dict): Dictionary of configuration parameters
    """
    ## Open Config JSON File
    with open(config_file,"r") as the_file:
        config = json.load(the_file)
    return config

def compile_results_metadata():
    """
    Compile all relevant files in a domain-transfer run

    Args:
        None
    
    Returns:
        results_meta (list of dict): List of metadata dictionaries
                            relevant for each individual domain transfer
                            experiment
    """
    ## Identify Results Directories
    results_dirs = glob(f"{DATA_DIR}/*{PREFIX}*")
    ## Create List of Results to Process
    results_meta = []
    for rd in results_dirs:
        rd_json = [g for g in glob(f"{rd}/*.json") if not g.endswith("splits.json")][0]
        results_meta.append({
                "directory":f"{rd}/",
                "config":load_config(rd_json),
                "predictions":f"{rd}/cross_validation/predictions.csv",
                "splits":f"{rd}/splits.json"
        })
    return results_meta

def summarize_performance(result_meta):
    """
    Compute classification performance scores within a domain-transfer
    experiment.

    Args:
        results_meta (dict): Dictionary of metadata relevant to a particular
                             domain transfer experiment
    
    Returns:
        results (pandas DataFrame): Fold/Group Level Classification Results
    """
    ## Metadata
    prediction_file = result_meta["predictions"]
    config = result_meta["config"]
    ## Load Predictions
    predictions = pd.read_csv(prediction_file, low_memory=False)
    ## Process Each Fold, Group
    results = []
    for group in predictions["group"].unique():
        for fold in predictions["fold"].unique():
            pred_subset = predictions.loc[(predictions["group"]==group)&
                                          (predictions["fold"]==fold)]
            ## ROC/AUC
            if len(set(pred_subset["y_pred"])) <= 2:
                fpr, tpr, thres = [], [], []
                auc = None
            else:
                fpr, tpr, thres = metrics.roc_curve(y_true=pred_subset["y_true"],
                                                    y_score=pred_subset["y_pred"])
                auc = metrics.auc(fpr, tpr)
            ## Precision/Recall/F1/Accuracy (By Class and Binary)
            p, r, f, s = metrics.precision_recall_fscore_support(y_true=pred_subset["y_true"],
                                                                 y_pred=(pred_subset["y_pred"]>0.5).astype(int),
                                                                 pos_label=1,
                                                                 average=None)
            pm, rm, fm, _ = metrics.precision_recall_fscore_support(y_true=pred_subset["y_true"],
                                                                    y_pred=(pred_subset["y_pred"]>0.5).astype(int),
                                                                    pos_label=1,
                                                                    average="binary")
            accuracy = metrics.accuracy_score(y_true=pred_subset["y_true"],
                                              y_pred=(pred_subset["y_pred"]>0.5).astype(int),)
            ## Record Results
            record = {"group":group,
                      "fold":fold,
                      "fpr_tpr_thres":(list(fpr), list(tpr), list(thres)),
                      "auc":auc,
                      "precision":list(p),
                      "binary_precision":pm,
                      "recall":list(r),
                      "binary_recall":rm,
                      "fscore":list(f),
                      "binary_fscore":fm,
                      "accuracy":accuracy,
                      "class_size":list(s),
                      "train_data":config.get("train_data"),
                      "test_data":config.get("test_data"),
                      "random_seed":config.get("random_seed"),
                      "stratified":config.get("stratified"),
                      }
            results.append(record)
    ## Format Results
    results = pd.DataFrame(results)
    return results

#######################
### Analysis
#######################

## Get Results Metadata
results_meta = compile_results_metadata()

## Summarize Performance
results_df = []
for r in results_meta:
    if not os.path.exists(r["predictions"]):
        continue
    results_df.append(summarize_performance(r))
results_df = pd.concat(results_df)

## Summary Statistics (Verify that all model random seeds were preserved by looking at train)
group = "dev"
summary_scores = {}
for met in ["binary_precision","binary_recall","binary_fscore","accuracy","auc"]:
    print("~"*15 + "\n" + met + "\n" + "~"*15)
    res = pd.pivot_table(results_df,
                         index = ["group","train_data"],
                         columns = ["test_data"],
                         values=met,
                         aggfunc=np.mean).loc[group]
    res = res.applymap(lambda x: "{:.2f}".format(x))
    print(res)
    summary_scores[met] = res

## Combine Results
datasets = ["clpsych_deduped","multitask","merged","rsdd","smhd","wolohan"]
datasets = [d for d in datasets if d in res.index.tolist()]
pretty_results = [[[] for _ in datasets] for __ in datasets]
for metric in ["binary_fscore","auc"]:
    met_res = summary_scores[metric]
    for d1, ds1 in enumerate(datasets):
        for d2, ds2 in enumerate(datasets):
            pretty_results[d1][d2].append(met_res.loc[ds1, ds2])
for r, row in enumerate(pretty_results):
    for c, col in enumerate(row):
        if "nan" not in col:
            pretty_results[r][c] = "/".join(col)
        else:
            pretty_results[r][c] = "--"
pretty_results = pd.DataFrame(pretty_results, index=datasets, columns=datasets)

##TODO: Model Analysis