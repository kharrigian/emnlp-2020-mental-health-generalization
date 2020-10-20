
"""
Primary modeling script
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

## Local Modules
from mhlib.model import train
from mhlib.model import cross_validation as cv

#################
### Globals
#################

## Output Path
RESULTS_DIR = "./data/results/domain_transfer/"

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
    parser.add_argument("--cache_models",
                        action="store_true",
                        default=False,
                        help="If included, will cache cross validation models")
    parser.add_argument("--cache_predictions",
                        action="store_true",
                        default=False,
                        help="If included, will cache fold predictions independently")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.config_filepath):
        raise ValueError(f"Could not find config file {args.config_filepath}")
    return args

def _across_domain_splits(config,
                          train_metadata_df,
                          test_metadata_df):
    """
    Creates K-fold splits within the training dataframe
    for hyperparameter optimization and leaves all options
    in the test_metadata_df as the test set.

    Args:
        config (dict): Experiment parameters
        train_metadata_df (pandas DataFrame): Metadata DataFrame for training data set
        test_metadata_df (pandas DataFrame): Metadata DataFrame for testing data set
    
    Returns:
        label_dictionaries (dict): Mapping between train, test, k-fold and (user file, label)
    """
    ## Random Seed
    np.random.seed(config["random_seed"])
    ## Make Copy of Metadata
    train_metadata_df = train_metadata_df.copy()
    test_metadata_df = test_metadata_df.copy()
    ## Get Within Domain Splits for Each Set Independently
    train_splits =  train.split_data(config,
                                     train_metadata_df,
                                     config["downsample"]["train"],
                                     config["downsample_size"]["train"],
                                     config["rebalance"]["train"],
                                     config["class_ratio"]["train"])
    test_splits = train.split_data(config,
                                   test_metadata_df,
                                   config["downsample"]["test"],
                                   config["downsample_size"]["test"],
                                   config["rebalance"]["test"],
                                   config["class_ratio"]["test"])
    ## Compile into Label Dictionaries
    label_dictionaries = {"train":{}, "test":{}}
    for fold in range(1, config["kfolds"]+1):
        label_dictionaries["train"][fold] = {
                "train":train_splits["train"][fold]["train"],
                "dev":test_splits["train"][fold]["dev"],
                "test":None
        }
    label_dictionaries["test"] = {1 : {
                "train":None,
                "dev":None,
                "test":test_splits["test"][1]["test"]
                                      }
                                  }
    return label_dictionaries


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
    test_dataset = config["test_data"]
    ## Set Random Seed (for sampling)
    np.random.seed(config["random_seed"])
    ## Case 1: Within-domain
    if train_dataset == test_dataset:
        metadata = train.load_dataset_metadata(train_dataset,
                                               config["target_disorder"],
                                               config["random_seed"])
        label_dictionaries = train.split_data(config,
                                              metadata,
                                              config["downsample"]["train"],
                                              config["downsample_size"]["train"],
                                              config["rebalance"]["train"],
                                              config["class_ratio"]["train"])
    ## Case 2: Across Domains
    else:
        train_metadata = train.load_dataset_metadata(train_dataset,
                                                     config["target_disorder"],
                                                     config["random_seed"])
        test_metadata = train.load_dataset_metadata(test_dataset,
                                                    config["target_disorder"],
                                                    config["random_seed"])
        label_dictionaries = _across_domain_splits(config,
                                                   train_metadata,
                                                   test_metadata)
    ## Cache Train/Test Splits
    outfile = "{}splits.json".format(config["outdir"])
    with open(outfile, "w") as the_file:
        json.dump(label_dictionaries, the_file)
    return label_dictionaries

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
                                      RESULTS_DIR,
                                      "domain_transfer")
    ## Create Train/Test Splits
    label_dictionaries = create_train_test_splits(config)
    ## Cross Validation
    cv_results = cv.run_cross_validation(config,
                                         label_dictionaries,
                                         args.cache_models,
                                         args.cache_predictions,
                                         train_date_boundaries=config.get("date_boundaries").get("train"),
                                         dev_date_boundaries=config.get("date_boundaries").get("test"),
                                         train_samples=config.get("post_sampling").get("train"),
                                         dev_samples=config.get("post_sampling").get("test"),
                                         drop_null_train=config.get("drop_null").get("train") if config.get("drop_null") else False,
                                         drop_null_dev=config.get("drop_null").get("test") if config.get("drop_null") else False)
    ## Evaluate Test Performance ## TODO
    if config["run_on_test"]:
        pass
    
#################
### Run
#################

if __name__ == "__main__":
    _ = main()