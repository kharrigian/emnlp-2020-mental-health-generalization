
#########################
### Imports
#########################

## Standard Library
import os
import sys

## External Libraries
import pandas as pd
import matplotlib.pyplot as plt

## Local
from .train import fit_model
from .plot import *
from ..util.logging import initialize_logger

#########################
### Globals
#########################

## Initialize Logger
LOGGER = initialize_logger()

#########################
### Functions
#########################

def format_cross_validation_predictions(target_disorder,
                                        kfold_results,
                                        label_dictionaries,
                                        outdir=None):
    """
    Format dictionaries of predictions from k-fold cross 
    validation into a single dataframe.

    Args:
        target_disorder (str): Name of the disorder being modeled (e.g. "depression")
        kfold_results (list of dict): List of experiment results. Output from run_cross_validation
        label_dictionaries (dict): Label splits by Train/Dev/Test
        outdir (str or None): If not None, caches predictions as a single CSV to this directory
    
    Returns:
        results_df (pandas DataFrame): DataFrame with [source, y_true, y_pred, fold, group] columns.
                        y_true will be binarized class labels. If available, y_pred will be
                        a predicted probability for the postive (target disorder) class.
    """
    ## Align Truth and Predictions
    results_df = []
    for _, fold_res in enumerate(kfold_results):
        ## Figure Out Which Fold is Being Shown
        fold = list(fold_res.keys())[0]
        ## Get Label/Pred Dicts
        fold_label_dict = label_dictionaries["train"][fold]
        fold_pred_dict = fold_res[fold]
        ## Get Index
        train_index = list(fold_label_dict["train"].keys())
        dev_index = list(fold_label_dict["dev"].keys())
        ## Create Fold DataFrame
        train_df = pd.DataFrame(index = train_index)
        train_df["y_true"] = train_df.index.map(lambda x: fold_label_dict["train"].get(x))
        train_df["y_pred"] = train_df.index.map(lambda x: fold_pred_dict["train"].get(x))
        train_df["group"] = "train"
        dev_df = pd.DataFrame(index = dev_index)
        dev_df["y_true"] = dev_df.index.map(lambda x: fold_label_dict["dev"].get(x))
        dev_df["y_pred"] = dev_df.index.map(lambda x: fold_pred_dict["dev"].get(x))
        dev_df["group"] = "dev"
        fold_df = train_df.append(dev_df)
        fold_df["fold"] = fold
        fold_df.reset_index().rename(columns={"index":"source"})
        results_df.append(fold_df)
    ## Concatenate
    results_df = pd.concat(results_df).reset_index(drop=True)
    results_df["y_true"] = results_df["y_true"].map(lambda i: 1 if i == target_disorder else 0 if i == "control" else -1)
    ## Fill Nulls With Uncertain Prediction
    results_df["y_pred"] = results_df["y_pred"].fillna(0.5)
    ## Save to Disk (If Desired)
    if outdir is not None:
        if not os.path.exists(outdir):
                os.makedirs(outdir)
        LOGGER.info("Saving Cross Validation Predictions to Disk")
        results_df.to_csv("{}predictions.csv".format(outdir), index=False)
    return results_df

def plot_results(outdir,
                 results_df):
    """
    Summarize K-fold Cross-Validation results using
    figures. Configuration should include "outdir" where
    the cross_validation subdirectory is cached with results

    Args:
        outdir (str): Path to output directory of plots
        results_df (pandas DataFrame): DataFrame with [source, y_true, 
                        y_pred, fold, group] columns.
    
    Returns:
        None, outputs summary information to disk.
    """
    ## Check Model Output Type (Class or Score)
    has_probs = np.any((results_df["y_pred"] != 0) | (results_df["y_pred"] != 1))
    ## Probability-inclusive Metrics
    if has_probs:
        ## ROC/AUC
        fig, ax = plot_roc_auc(results_df)
        fig.savefig("{}roc_auc.png".format(outdir))
        plt.close(fig)
        ## Predicted Probability Distribution
        fig, ax = plot_predicted_probability_distribution(results_df)
        fig.savefig("{}predicted_distributions.png".format(outdir))
        plt.close(fig)
    ## Precision/Recall/F1
    fig, ax = plot_precision_recall_f1(results_df)
    fig.savefig("{}precision_recall_f1.png".format(outdir))
    plt.close(fig)

def run_cross_validation(config,
                         label_dictionaries,
                         cache_models=False,
                         cache_predictions=True,
                         train_date_boundaries={},
                         dev_date_boundaries={},
                         train_samples={},
                         dev_samples={},
                         drop_null_train=False,
                         drop_null_dev=False):
    """
    Run K-Fold cross validation.

    Args:
        config (dict): Dictionary of experiment parameters (e.g. model and output directory)
        label_dictionaries (dict): Label splits by Train/Dev/Test
        cache_models (bool): If True, will cache models trained during each fold
        cache_predictions (bool): If True, will store individual prediction dicts of each fold
        train_date_boundaries (dict): {"min_date":val,"max_date":val}
        dev_date_boundaries (dict): {"min_date":val,"max_date":val}
        train_samples (dict): {"n_samples":val,"randomized":val}
        dev_samples (dict):  {"n_samples":val,"randomized":val}
        drop_null_train (bool): Whether to ignore samples without features during training
        drop_null_dev (bool): Whether to ignore samples without features during prediction

    Returns:
        cv_results (list of dict): List of predictions across the k-folds.
    """
    ## Cache of Results
    cv_outdir = "{}cross_validation/".format(config.get("outdir"))
    for k in range(config["kfolds"]):
        _ = os.makedirs("{}fold_{}/".format(cv_outdir, k+1))
    ## Run Through Folds
    cv_results = []
    for x, y in label_dictionaries["train"].items():
        LOGGER.info("Starting Fold {}/{}".format(x, config["kfolds"]))
        ## Fit Model and Make Predictions
        fold_outdir = "{}fold_{}/".format(cv_outdir, x)
        predictions, model = fit_model(config,
                                       data_key=x,
                                       train_labels=y["train"],
                                       dev_labels=y["dev"],
                                       output_dir=fold_outdir,
                                       grid_search=config["grid_search"],
                                       grid_search_kwargs=config["grid_search_kwargs"],
                                       cache_model=cache_models,
                                       cache_predictions=cache_predictions,
                                       train_date_boundaries=train_date_boundaries,
                                       dev_date_boundaries=dev_date_boundaries,
                                       train_samples=train_samples,
                                       dev_samples=dev_samples,
                                       drop_null_train=drop_null_train,
                                       drop_null_dev=drop_null_dev)
        ## Cache Predictions
        cv_results.append(predictions)
        ## Analyze Learned Features
        fig, ax = plot_model_coefficients(model, 30)
        if fig is not None:
            fig.savefig(f"{fold_outdir}model_diagnostics.png")
            plt.close(fig)
    ## Format Results
    results_df = format_cross_validation_predictions(target_disorder=config["target_disorder"],
                                                     kfold_results=cv_results,
                                                     label_dictionaries=label_dictionaries,
                                                     outdir=cv_outdir)
    ## Plot Performance Summary
    _ = plot_results(cv_outdir, results_df)
    LOGGER.info("Cross Validation Complete.")
