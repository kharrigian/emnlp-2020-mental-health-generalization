
"""
Evaluate a trained model on an established set of splits
"""

#######################
### Imports
#######################

## Standard Library
import os
import json
import gzip
import pprint
import argparse
from glob import glob
from datetime import datetime

## External Library
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

## Local
from mhlib.util.helpers import chunks
from mhlib.util.logging import initialize_logger

#######################
### Globals
#######################

## Logging
LOGGER = initialize_logger()
PRINTER = pprint.PrettyPrinter()

#######################
### Functions
#######################

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
    parser.add_argument("--model",
                        type=str,
                        help="Path to cached model file (.joblib)")
    parser.add_argument("--splits",
                        type=str,
                        help="Path to a cached splits.json file")
    parser.add_argument("--split_keys",
                        nargs=3,
                        type=str,
                        help="Keys to reference in splits. [train/test][fold][train/dev/test]")
    parser.add_argument("--output_folder",
                        type=str,
                        help="Where to cache predictions")
    parser.add_argument("--min_date",
                        type=str,
                        default="2000-01-01",
                        help="Lower date boundary (isoformat str) if desired")
    parser.add_argument("--max_date",
                        type=str,
                        default=None,
                        help="upper date boundary (isoformat str) if desired")
    parser.add_argument("--n_samples",
                        type=int,
                        default=None,
                        help="Number of post samples to isolate for modeling if desired")
    parser.add_argument("--randomized",
                        action="store_true",
                        default=False,
                        help="If included along with samples, will use randomized selection instead of recent")
    parser.add_argument("--drop_null",
                        action="store_true",
                        default=False,
                        help="If included, drop null rows from output")
    parser.add_argument("--chunksize",
                         type=int,
                         default=5000,
                         help="How many users to process at a time")
    parser.add_argument("--prob_threshold",
                        type=float,
                        default=0.5,
                        help="Binarization threshold")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Could not find model file {args.model}")
    if not os.path.exists(args.splits):
        raise FileNotFoundError(f"Could not find split file {args.splits}")
    return args

def get_date_bounds(args):
    """

    """
    min_date = pd.to_datetime(args.min_date)
    if args.max_date is None:
        max_date = pd.to_datetime(datetime.now())
    else:
        max_date = pd.to_datetime(args.max_date)
    return min_date, max_date

def score_predictions(y_true,
                      y_pred,
                      probability=False,
                      threshold=0.5):
    """
    Score predictive performance

    Args:
        y_true (1d-array): Ground truth labels (binary)
        y_pred (1-2d-array): Predicted classes or scores
        probability (bool): If True, expects scores to be passed
        threshold (float): Score threshold for classifiying as positive
    
    Returns:
        scores (dict): Classification scores
    """
    if probability:
        y_score = y_pred
        y_pred = (y_pred > threshold).astype(int)
    if np.all(y_pred == 0) or np.all(y_pred == 1):
        scores = {
            "precision":0,
            "recall":metrics.recall_score(y_true, y_pred, pos_label=1, average="binary"),
            "f1":0,
            "accuracy":metrics.accuracy_score(y_true, y_pred)
        }
    else:
        scores = {
            "precision":metrics.precision_score(y_true, y_pred, pos_label=1, average="binary"),
            "recall":metrics.recall_score(y_true, y_pred, pos_label=1, average="binary"),
            "f1":metrics.f1_score(y_true, y_pred, pos_label=1, average="binary"),
            "accuracy":metrics.accuracy_score(y_true, y_pred),
        }
    if probability:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        scores["auc"] = auc
        scores["roc"] = {"fpr":fpr,"tpr":tpr}
    return scores

def main():
    """

    """
    ## Parse Command-line Arguments
    args = parse_arguments()
    ## Prepare Output Folder
    output_folder = args.output_folder.rstrip("/") + "/"
    if not os.path.exists(output_folder):
        _ = os.makedirs(output_folder)
    ## Load Model
    LOGGER.info(f"Loading Model: {args.model}")
    model = joblib.load(args.model)
    ## Get Date Boundaries
    LOGGER.info(f"Parsing Date Boundaries")
    min_date, max_date = get_date_bounds(args)
    ## Load Splits
    with open(args.splits,"r") as the_file:
        splits = json.load(the_file)
    ## Isolate Appropriate Splits
    sample_group, fold, train_group = args.split_keys
    splits = splits[sample_group][fold][train_group]
    ## Chunk Files
    filenames = sorted(splits.keys())
    file_chunks = list(chunks(filenames, args.chunksize))
    ## Make Predictions in Chunks
    y_pred = dict()
    for f, file_chunk in enumerate(file_chunks):
        LOGGER.info("Making Predictions for File Chunk {}/{}".format(f+1, len(file_chunks)))
        f_pred = model.predict(file_chunk,
                               min_date=min_date,
                               max_date=max_date,
                               n_samples=args.n_samples,
                               randomized=args.randomized,
                               drop_null=args.drop_null)
        y_pred.update(f_pred)
    ## Combine Predictions with Ground Truth
    y_pred = pd.concat([pd.Series(splits).to_frame("y_true"),
                        pd.Series(y_pred).to_frame("y_pred")],
                       axis=1,
                       sort=True)
    y_pred["y_true"] = y_pred["y_true"].map(lambda i: int(i != "control"))
    y_pred.index = y_pred.index.map(os.path.abspath)
    ## Cache Predictions
    pred_file = f"{output_folder}{model._target_disorder}.predictions.csv"
    y_pred.to_csv(pred_file)
    ## Score Predictions
    has_probability = isinstance(y_pred["y_pred"].values[0], float)
    scores = score_predictions(y_pred["y_true"].values,
                               y_pred["y_pred"].values,
                               probability=has_probability,
                               threshold=args.prob_threshold)
    if has_probability:
        for curve in ["fpr","tpr"]:
            scores["roc"][curve] = list(scores["roc"][curve])
    ## Cache Scores
    score_file = f"{output_folder}{model._target_disorder}.scores.json"
    with open(score_file,"w") as the_file:
        json.dump(scores, the_file)
    ## Show Scores
    scores_formatted = PRINTER.pformat(scores)
    LOGGER.info("~"*20 + "Scores" + "~"*20)
    LOGGER.info(scores_formatted)
    ## Update User
    LOGGER.info(f"Cached predictions and scores in: {output_folder}")
    LOGGER.info("Script Complete!")

#######################
### Execute
#######################

if __name__ == "__main__":
    _ = main()