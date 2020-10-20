
"""
General Script for Training a Mental Health Classifier
"""

#################
### Imports
#################

## Standard Libary
import os
import json
import gzip
import argparse

## External Libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

## Local Modules
from mhlib.model import plot
from mhlib.model import train
from mhlib.model import cross_validation as cv
from mhlib.util.logging import initialize_logger

## Create Logger
LOGGER = initialize_logger()

#################
### Functions
#################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.

    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Run modeling experiments")
    ## Generic Arguments
    parser.add_argument("config_filepath",
                        type=str,
                        help="Path to your configuration JSON file")
    parser.add_argument("--cross_validation",
                        action="store_true",
                        default=False,
                        help="If included, run cross validation before training a full model")
    parser.add_argument("--cache_cross_validation_models",
                        action="store_true",
                        default=False,
                        help="If included, will cache cross validation models")
    parser.add_argument("--cache_cross_validation_predictions",
                        action="store_true",
                        default=False,
                        help="If included, will cache fold predictions independently")
    parser.add_argument("--evaluate_test",
                        action="store_true",
                        default=False,
                        help="If included, the model will make predictions on the test set")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.config_filepath):
        raise ValueError(f"Could not find config file {args.config_filepath}")
    if (args.cache_cross_validation_models or args.cache_cross_validation_predictions) and not args.cross_validation:
        raise ValueError("Asked to cache cross validation outputs without turning on cross validation.")
    return args

def create_train_test_splits(config):
    """
    Create training and testing splits for the experiment,
    preserving random sampling.

    Args:
        config (dict): Dictionary of experiment parameters
    
    Returns:
        label_dictionaries (dict): Train/Dev/Test Label Dictionaries
    """
    ## Identify Datasets
    train_dataset = config["train_data"]
    ## Set Random Seed (for sampling)
    np.random.seed(config["random_seed"])
    ## Load Labels
    metadata = train.load_dataset_metadata(train_dataset,
                                           target_disorder=config["target_disorder"],
                                           random_state=config["random_seed"])
    ## Split Data into Folds
    label_dictionaries = train.split_data(config,
                                          metadata,
                                          config["downsample"]["train"],
                                          config["downsample_size"]["train"],
                                          config["rebalance"]["train"],
                                          config["class_ratio"]["train"],
                                          downsample_test=config["downsample"].get("test"),
                                          downsample_size_test=config["downsample_size"].get("test"),
                                          rebalance_test=config["rebalance"].get("test"),
                                          class_ratio_test=config["class_ratio"].get("test"))
    ## Cache Train/Test Splits
    outfile = "{}splits.json".format(config["outdir"])
    with open(outfile, "w") as the_file:
        json.dump(label_dictionaries, the_file)
    return label_dictionaries

def separate_train_test_dict(label_dictionaries):
    """

    """
    train_labels = {}
    for _, y in label_dictionaries["train"].items():
        train_labels.update(y["dev"])
    test_labels = label_dictionaries["test"][1]["test"]
    return train_labels, test_labels

def format_predictions(train_predictions,
                       test_predictions,
                       train_labels,
                       test_labels):
    """

    """
    ## Format Into DataFrames
    train_df = pd.concat([pd.Series(train_labels), 
                          pd.Series(train_predictions)],
                         sort=True,
                         axis=1).rename(columns={0:"y_true",1:"y_pred"})
    test_df = pd.concat([pd.Series(test_labels),
                         pd.Series(test_predictions)],
                        sort=True,
                        axis=1).rename(columns={0:"y_true",1:"y_pred"})
    train_df["group"] = "train"
    test_df["group"] = "test"
    ## Concatenate Predictions
    results_df = pd.concat([train_df, test_df])
    results_df["fold"] = 1
    results_df = results_df.reset_index().rename(columns={"index":"source"})
    results_df["y_true"] = results_df["y_true"].map(lambda i: 0 if i == "control" else 1)
    ## Fill Missing With Uncertain Prediction
    results_df["y_pred"] = results_df["y_pred"].fillna(0.5)
    return results_df

def _plot_confusion_matrix(results_df):
    """

    """
    fig, ax = plt.subplots(1, 2, figsize=(10,5.8))
    for g, group in enumerate(["train","test"]):
        group_data = results_df.loc[results_df["group"]==group]
        cm = confusion_matrix(group_data["y_true"],
                              group_data["y_pred"]>=0.5,
                              labels=[0,1])
        cm_normalized = cm / cm.sum(axis=1,keepdims=True)
        ax[g].imshow(cm_normalized,
                     aspect="auto",
                     interpolation="nearest",
                     cmap=plt.cm.Purples)
        for i, row in enumerate(cm_normalized):
            for j, val in enumerate(row):
                ax[g].text(j, i, "{:,d}".format(cm[i,j]), ha="center", va="center",
                           color="black" if val < 0.5 else "white")
        ax[g].set_xticks([0,1])
        ax[g].set_xticklabels(["Control","Target"],rotation=45,ha="right")
        ax[g].set_yticks([0,1])
        ax[g].set_ylim(1.5,-0.5)
        ax[g].set_yticklabels(["Control","Target"])
        ax[g].set_title(group.title(), loc="left", fontsize=10, fontweight="bold")
        ax[g].set_xlabel("Predicted", fontweight="bold")
        ax[g].set_ylabel("True", fontweight="bold")
    fig.tight_layout()
    return fig, ax

def plot_results(results_df,
                 outdir):
    """

    """
    ## Check Model Output Type (Class or Score)
    has_probs = np.any((results_df["y_pred"] != 0) | (results_df["y_pred"] != 1))
    ## Probability-inclusive Metrics
    if has_probs:
        ## ROC/AUC
        fig, ax = plot.plot_roc_auc(results_df,
                                    groups=["train","test"])
        fig.savefig("{}roc_auc.png".format(outdir), dpi=150)
        plt.close(fig)
        ## Predicted Probability Distribution
        fig, ax = plot.plot_predicted_probability_distribution(results_df,
                                                               groups=["train","test"])
        fig.savefig("{}predicted_distributions.png".format(outdir), dpi=150)
        plt.close(fig)
    ## Precision/Recall/F1
    fig, ax = plot.plot_precision_recall_f1(results_df,
                                            groups=["train","test"])
    fig.savefig("{}precision_recall_f1.png".format(outdir), dpi=150)
    ## Confusion Matrix
    fig, ax = _plot_confusion_matrix(results_df)
    fig.savefig("{}confusion_matrix.png".format(outdir), dpi=150)
    plt.close(fig)

def main():
    """
    Run the domain transfer experiment from start to finish.

    Args:
        None
    
    Returns:
        None
    """
    ## Parse command-line arguments
    args = parse_arguments()
    ## Load Configuration, Save Cached Version for Reference
    config = train.load_configuration(args,
                                      train.MODELS_DIR,
                                      "model")
    ## Create Train/Test Splits
    label_dictionaries = create_train_test_splits(config)
    ## Cross Validation
    if args.cross_validation:
        cv_results = cv.run_cross_validation(config,
                                             label_dictionaries,
                                             args.cache_cross_validation_models,
                                             args.cache_cross_validation_predictions,
                                             train_date_boundaries=config.get("date_boundaries").get("train"),
                                             dev_date_boundaries=config.get("date_boundaries").get("train"),
                                             drop_null_train=config.get("drop_null").get("train") if config.get("drop_null") else False,
                                             drop_null_dev=config.get("drop_null").get("train") if config.get("drop_null") else False)
    ## Train Full Model
    train_labels, test_labels = separate_train_test_dict(label_dictionaries)
    predictions, model = train.fit_model(config=config,
                                         data_key=config.get("experiment_name"),
                                         train_labels=train_labels,
                                         dev_labels=None,
                                         output_dir=config.get("outdir"),
                                         grid_search=config.get("grid_search"),
                                         grid_search_kwargs=config.get("grid_search_kwargs"),
                                         cache_model=True,
                                         cache_predictions=True,
                                         train_date_boundaries=config.get("date_boundaries").get("train"),
                                         train_samples=config.get("post_sampling").get("train"),
                                         drop_null_train=config.get("drop_null").get("train") if config.get("drop_null") else False,
                                         drop_null_dev=config.get("drop_null").get("train") if config.get("drop_null") else False)
    ## Plot Model Diagnostics (e.g. Feature Coefficients)
    fig, ax = plot.plot_model_coefficients(model, 30)
    if fig is not None:
        fig.savefig("{}model_diagnostics.png".format(config.get("outdir")), dpi=150)
        plt.close(fig)
    ## Test Set Evaluation
    if args.evaluate_test:
        ## Make Predictions
        predictions[config.get("experiment_name")]["test"] = model.predict(sorted(test_labels),
                                                                           min_date=config.get("date_boundaries").get("test").get("min_date"),
                                                                           max_date=config.get("date_boundaries").get("test").get("max_date"),
                                                                           n_samples=config.get("post_sampling").get("test").get("n_samples"),
                                                                           randomized=config.get("post_sampling").get("test").get("randomized"),
                                                                           drop_null=True)
        _ = predictions[config.get("experiment_name")].pop("dev", None)
        ## Format Results
        results_df = format_predictions(train_predictions=predictions[config.get("experiment_name")]["train"],
                                        test_predictions=predictions[config.get("experiment_name")]["test"],
                                        train_labels=train_labels,
                                        test_labels=test_labels)
        ## Plot Performance Measures
        _ = plot_results(results_df,
                         config.get("outdir"))
    ## Cache Predictions
    pred_file = "{}predictions.tar.gz".format(config.get("outdir"))
    with gzip.open(pred_file, "wt", encoding="utf-8") as the_file:
        json.dump(predictions, the_file)
    LOGGER.info("Script Complete.")

########################
### Execution
########################

if __name__ == "__main__":
    _ = main()
