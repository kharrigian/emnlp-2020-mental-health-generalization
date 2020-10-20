
"""
Helper script ran as an independent job. Scheduled using temporal_effects.py
"""

#####################
### Imports
#####################

## Standard Library
import os
import gzip
import json
import argparse

## External Libraries
import joblib

## Local
from mhlib.model import train
from mhlib.util.logging import initialize_logger

#####################
### Globals
#####################

## Logger
LOGGER = initialize_logger()

## Root Repository Directory
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)) + "/../../../"

## Output Directory
RESULTS_DIR = f"{ROOT_PATH}data/results/temporal_effects/"

#####################
### Functions
#####################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.

    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Run temporal effects subprocess.")
    ## Generic Arguments
    parser.add_argument("--config",
                        type=str,
                        help="Path to experiment config file")
    parser.add_argument("--target_date",
                        type=str,
                        help="Target Disorder Training Date")
    parser.add_argument("--control_date",
                        type=str,
                        help="Control Group Training Date")
    parser.add_argument("--splits",
                        type=str,
                        help="Path to split file")
    parser.add_argument("--fold",
                        type=int,
                        help="Cross Validation Fold")
    parser.add_argument("--cache_models",
                        action="store_true",
                        default=False,
                        help="Store cross validation model")
    parser.add_argument("--cache_predictions",
                        action="store_true",
                        default=False,
                        help="Store cross validation predictions")
    parser.add_argument("--models_dir",
                        type=str,
                        default=None,
                        help="Path to pre-trained model cache directory (optional)")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.config):
        raise ValueError(f"Could not find config file {args.config}")
    if not os.path.exists(args.splits):
        raise ValueError(f"Could not find split file {args.splits}")
    return args

def fit_model(config,
              target_train_date,
              control_train_date,
              splits,
              fold,
              cache_model=False,
              cache_predictions=False):
    """
    Fit a mental health classifier using data from temporal effects experiment

    Args:
        config (dict): Experiment configuration
        target_train_date (str): Source domain train period for target disorder
        control_train_date (str): Source domain train period for control group
        splits (dict): Data splits
        fold (int): Which fold is being fit
        cache_model (bool): Whether classifier should be saved to disk
        cache_predictions (bool): Whether training predictions should be saved to disk
    
    Returns:
        predictions (dict): Training predictions
        model (MentalHealthClassifier): Fit model object
        model_outdir (str): Where the model has been saved
    """
    ## Isolate Appropriate Training Data
    target_train_labels = splits["train"][(target_train_date, control_train_date)][fold][config["target_disorder"]]
    control_train_labels = splits["train"][(target_train_date, control_train_date)][fold]["control"]
    train_labels = {**target_train_labels, **control_train_labels}
    ## Create Output Directory
    model_outdir = "{}cross_validation/{}_{}_{}/".format(config.get("outdir"), target_train_date, control_train_date, fold)
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)
    ## Fit Model
    predictions, model = train.fit_model(config,
                                         data_key="{}_{}_{}".format(target_train_date, control_train_date, fold),
                                         train_labels=train_labels,
                                         dev_labels=None,
                                         output_dir="{}",
                                         grid_search=config["grid_search"],
                                         grid_search_kwargs=config["grid_search_kwargs"],
                                         cache_model=cache_model,
                                         cache_predictions=cache_predictions,
                                         train_samples=config.get("post_sampling").get("train"),
                                         drop_null_train=config.get("drop_null").get("train") if config.get("drop_null") else False,
                                         drop_null_dev=config.get("drop_null").get("test") if config.get("drop_null") else False) 
    ## Return Predictions and Model
    return predictions, model, model_outdir

def load_pretrained_model(args,
                          config):
    """

    """
    ## Check Input
    if args.models_dir is None:
        return None, None
    ## Specify Model Identifiers
    model_path = "{}{}_{}_{}_{}.joblib".format(args.models_dir,
                                               config.get("train_data"),
                                               args.target_date,
                                               args.control_date,
                                               args.fold)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = None
    return model, model_path

def apply_model_to_train(config,
                         target_train_date,
                         control_train_date,
                         splits,
                         fold,
                         model):
    """

    """
    ## Isolate Appropriate Training Data
    target_train_labels = splits["train"][(target_train_date, control_train_date)][fold][config["target_disorder"]]
    control_train_labels = splits["train"][(target_train_date, control_train_date)][fold]["control"]
    train_labels = {**target_train_labels, **control_train_labels}
    ## Create Output Directory
    model_outdir = "{}cross_validation/{}_{}_{}/".format(config.get("outdir"), target_train_date, control_train_date, fold)
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)
    ## Make Predictions
    train_predictions = model.predict(list(train_labels.keys()),
                                      n_samples=config.get("post_sampling").get("train").get("n_samples"),
                                      randomized=config.get("post_sampling").get("train").get("randomized"),
                                      drop_null=config.get("drop_null").get("train") if config.get("drop_null") else False)
    ## Format
    model_id = "{}_{}_{}".format(target_train_date, control_train_date, fold)
    train_predictions = {model_id:{"train":train_predictions}}
    return train_predictions, model_outdir

def apply_model_to_dev(config,
                       splits,
                       fold,
                       model):
    """
    Apply a mental-health classifier to development date splits

    Args:
        config (dict): Experiment configuration
        splits (dict): Cross validation splits
        fold (int): Which fold of cross validation to apply model to
        model (MentalHealthClassifier): Trained model object
    
    Returns:
        dev_predictions (dict): Predictions on development data for each time split
    """
    ## Identify Development Dates
    dev_dates = list(splits["dev"].keys())
    ## Prediction Cache
    dev_predictions = {}
    ## Cycle Through Development Dates
    for d, dd in enumerate(dev_dates):
        LOGGER.info("Applying Model to Date Range {}/{} ({})".format(d+1, len(dev_dates), dd))
        ## Isolate Appropriate Files
        dd_control_labels = splits["dev"][dd][fold]["control"]
        dd_target_labels = splits["dev"][dd][fold][config["target_disorder"]]
        dd_labels = {**dd_control_labels, **dd_target_labels}
        ## Make Predictions
        dd_predictions = model.predict(list(dd_labels.keys()),
                                       n_samples=config.get("post_sampling").get("test").get("n_samples"),
                                       randomized=config.get("post_sampling").get("test").get("randomized"),
                                       drop_null=config.get("drop_null").get("test") if config.get("drop_null") else False)
        dev_predictions[dd] = dd_predictions
    ## Return
    return dev_predictions


def main():
    """
    Train a mental health classifier on a particular 
    time split and apply to others. Expected subprocess
    of temporal_effects.py, not a standalone script.

    Args:
        None
    
    Returns:
        None
    """
    ## Parse Command-Line Arguments
    args = parse_arguments()
    ## Load Configuration
    LOGGER.info("Loading Configuration")
    with open(args.config,"r") as the_file:
        config = json.load(the_file)
    ## Load Splits
    LOGGER.info("Loading Splits")
    splits = joblib.load(args.splits)
    ## Look for Pre-trained Model, Identify Model Path
    model, model_path = load_pretrained_model(args,
                                              config)
    ## Based on Existence of Model, Fit or Apply
    if model_path is not None and os.path.exists(model_path):
        ## Load and Apply Previous Model
        LOGGER.info(f"Loaded model from path: {model_path}")
        predictions, model_outdir = apply_model_to_train(config=config,
                                                         target_train_date=args.target_date,
                                                         control_train_date=args.control_date,
                                                         splits=splits,
                                                         fold=args.fold,
                                                         model=model)
    else:
        ## Fit New Classifier or Apply Old to Training Set
        LOGGER.info("Starting New Fit Procedure")
        predictions, model, model_outdir = fit_model(config=config,
                                                     target_train_date=args.target_date,
                                                     control_train_date=args.control_date,
                                                     splits=splits,
                                                     fold=args.fold,
                                                     cache_model=args.cache_models,
                                                     cache_predictions=args.cache_predictions)
    ## Apply Classifier to Each Development Set, Store Predictions
    LOGGER.info("Begining Model Application on Development Data")
    development_predictions = apply_model_to_dev(config=config,
                                                 splits=splits,
                                                 fold=args.fold, 
                                                 model=model)
    ## Combine Predictions
    model_id = model_outdir.split("/")[-2]
    predictions[model_id]["dev"] = development_predictions
    ## Cache Predictions
    LOGGER.info("Caching Predictions")
    with gzip.open(f"{model_outdir}predictions.json.gz", "wt") as the_file:
        json.dump(predictions, the_file)
    ## Cache Model
    if args.cache_models:
        LOGGER.info("Caching Model")
        _ = model.dump(f"{model_outdir}model.joblib")
    ## Cache Temp Model
    if args.models_dir is not None and not os.path.exists(model_path):
        _ = model.dump(f"{model_path}")
    ## Script Complete
    LOGGER.info("Script Complete!")

#####################
### Execution
#####################

if __name__ == "__main__":
    _ = main()