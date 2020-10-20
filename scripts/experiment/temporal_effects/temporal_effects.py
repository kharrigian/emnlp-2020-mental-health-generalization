
"""
Explore model performance as a function of different temporal training/test splits
"""

##################
### Imports
##################

## Standard Library
import os
import sys
import gzip
import json
import argparse
import subprocess
from time import sleep
from glob import glob
from itertools import product
from functools import partial
from datetime import datetime
from collections import Counter

## External Libraries
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack
from sklearn import metrics
from sklearn.model_selection import (KFold,
                                     StratifiedKFold)

## Local
from mhlib.model import train
from mhlib.util.logging import initialize_logger
from mhlib.util.multiprocessing import MyPool as Pool
from mhlib.model.data_loaders import LoadProcessedData

##################
### Globals
##################

## Logger
LOGGER = initialize_logger()

## Root Repository Directory
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)) + "/../../../"

## Output Directory
RESULTS_DIR = f"{ROOT_PATH}data/results/temporal_effects/"

## CLSP Grid Information
USERNAME = "kharrigian"

##################
### Functions
##################

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
                        help="If included, will cache models")
    parser.add_argument("--cache_predictions",
                        action="store_true",
                        default=False,
                        help="If included, will cache predictions independently")
    parser.add_argument("--train_date_distribution",
                        type=str,
                        default=None,
                        help="If it exists, load a pre-compiled training date distribution")
    parser.add_argument("--test_date_distribution",
                        type=str,
                        default=None,
                        help="If it exists, load a pre-compiled testing date distribution")
    parser.add_argument("--models_dir",
                        type=str,
                        default=None,
                        help="If exists, try to find pre-trained model. Otherwise, create new one.")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.config_filepath):
        raise FileNotFoundError(f"Could not find config file {args.config_filepath}")
    if args.train_date_distribution is not None and not os.path.exists(args.train_date_distribution):
        raise FileNotFoundError(f"Could not find specified training date distribution {args.train_date_distribution}")
    if args.test_date_distribution is not None and not os.path.exists(args.test_date_distribution):
        raise FileNotFoundError(f"Could not find specified training date distribution {args.test_date_distribution}")
    if args.models_dir is not None and not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)
    return args

def assign_date_values_to_bins(values,
                               bins):
    """
    Args:
        values (dates sorted new to old)
        bins (date bins sorted old to new)
    
    Returns:
        assignments (list): List of bin assignments per timestamp
    """
    ## Reverse Values
    values = values[::-1]
    ## Initialize Index Variables
    b = 0
    i = 0
    ## Cycle Trhrough Values
    n = len(values)
    m = len(bins)
    ## Initialize Cache
    assignments = []
    ## Move Indices to Boundaries
    while i < n and values[i] < bins[0]:
        i += 1
        assignments.append(None)
    while b < m and i < n and values[i] >= bins[b]:
        b += 1
    b -= 1
    while i < n:
        if b == m - 1:
            assignments.append(b)
            i += 1
        else:
            if values[i] >= bins[b] and values[i] < bins[b+1]:
                assignments.append(b)
                i += 1
            else:
                b += 1
    ## Check Assignments
    assert len(assignments) == n
    return assignments

def load_timestamp_vector(filename,
                          data_loader,
                          date_range_bins):
    """
    Load distribution of posts over time within a file

    Args:
        filename (str): Path to processed data file
        data_loader (object): Data loader object
        date_range_bins (list of datetimes): Date bin boundaries
    
    Returns:
        filename (str): Input filename
        x (csr_matrix): Sparse vector representing post distribution over time
    """
    ## Loader User Data
    user_data = data_loader.load_user_data(filename)
    ## No Data after Filtering
    if len(user_data) == 0:
        return filename, csr_matrix(np.zeros(len(date_range_bins)))
    ## Get Timestamps
    timestamps = list(map(lambda u: datetime.fromtimestamp(u["created_utc"]), user_data))
    ## Assign To Bins
    timestamp_bins = assign_date_values_to_bins(timestamps, date_range_bins)
    ## Filter Out Null Bins
    timestamp_bins = list(filter(lambda tb: tb is not None, timestamp_bins))
    ## Count Distribution
    timestamp_count = Counter(timestamp_bins)
    ## Create Vector
    x = np.zeros(len(date_range_bins))
    for tcb, tcc in timestamp_count.items():
        x[tcb] = tcc
    x = csr_matrix(x)
    return filename, x

def retrieve_temporal_breakdown(metadata,
                                config,
                                data_split="train"):
    """
    Load distribution of posts over time for all users in a metadata fil

    Args:
        metadata (pandas DataFrame): Label file for a dataset
        config (dict): Experiment configuration
    
    Returns:
        filenames (list): List of filenames for each row in time bin array
        X (csr_matrix): Time bin array for users in metadata
    """
    ## Data Loader
    data_loader = LoadProcessedData(filter_mh_subreddits=config["vocab_kwargs"]["filter_mh_subreddits"],
                                    filter_mh_terms=config["vocab_kwargs"]["filter_mh_terms"],
                                    max_documents_per_user=None)
    ## Date Range
    date_range_bins = pd.to_datetime(config["date_boundaries"][data_split])
    ## Format Vectorization Function
    mp_timestamp_vectorizer = partial(load_timestamp_vector,
                                      data_loader=data_loader,
                                      date_range_bins=date_range_bins)
    ## Vectorize Data
    mp = Pool(config["jobs"])
    res = list(tqdm(mp.imap_unordered(mp_timestamp_vectorizer, metadata["source"].tolist()),
                    total=len(metadata),
                    desc="Timestamp Loader",
                    file=sys.stdout))
    mp.close()
    ## Parse Vectors
    filenames = [r[0] for r in res]
    X = vstack([r[1] for r in res])
    return filenames, X

def create_temporal_splits(metadata,
                           filenames,
                           X_time,
                           config,
                           data_split="train",
                           increment_sample_seed=True):
    """
    Create temporally-aligned cross validation splits that meet
    criteria regarding user set size specified in experiment configuration file

    Args:
        metadata (pandas DataFrame): Label data for a particiular data set
        filenames (list): List of filenames associated with rows in time bin array
        X_time (csr_matrix): Distribution of posts over time for different users
        config (dict): Experiment configuration parameters
        data_split (str): "train" or "test" (e.g. source vs. target domain)
        increment_sample_seed (bool): If True, adds one to seed used for sampling valid user
                                      subset to promote sample diversity
        
    Returns:
        label_dictionaries (dict): Dictionary mapping splits, folds, time periods
        time_boundaries  (dict): Time identifier str to min/max dates in a period
    """
    ## Isolate Dependent Variable
    y = metadata.set_index("source").loc[filenames][config["target_disorder"]].values
    ## Sum Over Classes
    post_threshold = config["min_posts"][data_split]
    y_time_sum = vstack([csr_matrix((X_time[y=="control"]>=post_threshold).sum(axis=0)),
                         csr_matrix((X_time[y==config["target_disorder"]]>=post_threshold).sum(axis=0))]).toarray()
    ## Identify Time Bins Meeting Threshold
    valid_time_bins = list(np.nonzero((y_time_sum >= config["min_users_per_split"]).all(axis=0))[0])
    invalid_time_bins = list(np.nonzero((y_time_sum < config["min_users_per_split"]).any(axis=0))[0])
    ## Strict Boundaries
    max_date = config["date_boundaries"][data_split][-1]
    min_date = config["date_boundaries"][data_split][0]
    ## Valid Date Boundaries
    time_boundaries = {}
    out_of_bounds = []
    for i in valid_time_bins:
        if i == len(config["date_boundaries"][data_split]) - 1:
            bounds = (config["date_boundaries"][data_split][i], datetime.now().date().isoformat())
        else:
            bounds = (config["date_boundaries"][data_split][i], config["date_boundaries"][data_split][i+1])
        if bounds[0] < min_date:
            out_of_bounds.append(i)
            continue
        if bounds[1] > max_date:
            out_of_bounds.append(i)
            continue
        time_boundaries[config["date_boundaries"][data_split][i]] = bounds
    for o in out_of_bounds:
        valid_time_bins.remove(o)
        invalid_time_bins.append(o)
    valid_time_bins = sorted(valid_time_bins)
    invalid_time_bins = sorted(invalid_time_bins)
    ## Data Split Balancing
    users_per_split = y_time_sum
    if config["balance_date_splits"]:
        users_per_split = np.ones_like(y_time_sum) * np.floor(np.min(y_time_sum[:, valid_time_bins]) * (1-config["test_size"]))
    users_per_split[:, invalid_time_bins] = 0
    users_per_split = users_per_split.astype(int)
    ## Create Label Dictionaries
    valid_time_boundaries = [config["date_boundaries"][data_split][v] for v in valid_time_bins]
    label_dictionaries = {v:{} for v in valid_time_boundaries}
    ## Increment Sample Seed Each Time to Promote Sample User Diversity
    sample_seed = config["random_seed"]
    ## Cycle Through Bins
    for i, v in zip(valid_time_bins, valid_time_boundaries):
        ## Identify Valid Source Users
        valid_bin_users = (X_time[:,i] >= post_threshold).toarray().T[0]
        ## Isolate Valid Metadata
        valid_metadata = metadata.set_index("source").loc[np.array(filenames)[valid_bin_users]].reset_index().copy()
        ## Sample Test Set (Balanced)
        test_depression = valid_metadata.loc[valid_metadata[config["target_disorder"]]==config["target_disorder"]].sample(
                            frac = config["test_size"],
                            replace = False,
                            random_state = sample_seed)
        test_control = valid_metadata.loc[valid_metadata[config["target_disorder"]]=="control"].sample(
                            n = len(test_depression),
                            replace = False,
                            random_state = sample_seed)
        test_set = test_depression.append(test_control)
        ## Split Remaining Data
        remaining_data = valid_metadata.loc[~valid_metadata["source"].isin(test_set["source"])]
        ## Rebalancing (Train and Dev Only)
        if config["rebalance"][data_split]:
            remaining_data = train._rebalance(remaining_data,
                                              config["target_disorder"],
                                              config["class_ratio"][data_split],
                                              config["random_seed"])
        ## Downsample
        remaining_target_group = remaining_data.loc[remaining_data[config["target_disorder"]]==config["target_disorder"]]
        remaining_control_group = remaining_data.loc[remaining_data[config["target_disorder"]]=="control"]
        if config["downsample"][data_split]:
            remaining_control_group = remaining_control_group.sample(n=min(config["downsample_size"][data_split], users_per_split[0][i]),
                                                                     random_state=sample_seed,
                                                                     replace=False)    
            remaining_target_group = remaining_target_group.sample(n=min(config["downsample_size"][data_split], users_per_split[1][i]),
                                                                   random_state=sample_seed,
                                                                   replace=False)
        elif config["balance_date_splits"]:
            remaining_control_group = remaining_control_group.sample(n=users_per_split[0][i],
                                                                     random_state=sample_seed,
                                                                     replace=False)    
            remaining_target_group = remaining_target_group.sample(n=users_per_split[1][i],
                                                                   random_state=sample_seed,
                                                                   replace=False)
        ## Combine Data
        remaining_data = pd.concat([remaining_control_group, remaining_target_group]).reset_index(drop=True).copy()
        ## Create Devs Split
        if config["stratified"]:
            splitter = StratifiedKFold(n_splits=config["kfolds"],
                                       shuffle=True,
                                       random_state=config["random_seed"])
        else:
            splitter = KFold(n_splits=config["kfolds"],
                             shuffle=True,
                             random_state=config["random_seed"])
        splits = splitter.split(remaining_data.index.tolist(),
                                remaining_data[config["target_disorder"]].values)
        ## Create Split Dictionaries
        train_splits = {}
        for k, (train_, dev_) in enumerate(splits):
            train_splits[k+1] = {
                "train": remaining_data.loc[train_].set_index("source")[config["target_disorder"]].to_dict(),
                "dev": remaining_data.loc[dev_].set_index("source")[config["target_disorder"]].to_dict(),
                "test": None,
                    }
        test_splits = {1:  {
                            "train": None,
                            "dev": None,
                            "test": test_set.set_index("source")[config["target_disorder"]].to_dict()
                            }
                      }
        ## Create Standardized Label Dictionaries
        label_dictionaries[v] = {
                        "train":train_splits,
                        "test":test_splits
        }
        ## Increment Sample Seed
        if increment_sample_seed:
            sample_seed += 1
    return label_dictionaries, time_boundaries


def merge_dictionaries(train_label_dictionaries,
                       test_label_dictionaries,
                       config):
    """
    Combine label dictionaries from source/target domains into a single
    label dictionary

    Args:
        train_label_dictionaries (dict): Label dictionaries in source domain
        test_label_dictionaries (dict): Label dictionaries in target domain
        config (dict): Experiment configuration parameters
    
    Returns:
        label_dictionaries (dict): Merged label dictionary
        train_date_combinations (list of tuple): Training period combinations (target disorder, control)
        train_date_ranges (list): List of time periods in source domain
        test_date_ranges (list): List of time periods in target domain
    """
    ## Overlap of Date Ranges
    train_date_ranges = sorted(train_label_dictionaries.keys())
    test_date_ranges = sorted(test_label_dictionaries.keys())
    if config["mixed_time_windows"]:
        train_date_combinations = list(product(train_date_ranges, train_date_ranges))
    else:
        train_date_combinations = [(t, t) for t in train_date_ranges]
    ## Cross Validation Folds
    folds = list(range(1, config["kfolds"]+1))
    ## Merge Dictionaries
    label_dictionaries = {
                            "train":{c:{f:{} for f in folds} for c in train_date_combinations},
                            "dev":{td:{f:{} for f in folds} for td in test_date_ranges},
                            "test":{td:{1:{}} for td in test_date_ranges},
    }
    ## Cycle Through Combinations
    target_disorder = config["target_disorder"]
    for target_date, control_date in train_date_combinations:
        for fold in folds:
            ## Training
            target_train = dict((x,y) for x,y in train_label_dictionaries[target_date]["train"][fold]["train"].items() if y!="control")
            control_train = dict((x,y) for x,y in train_label_dictionaries[control_date]["train"][fold]["train"].items() if y=="control")
            label_dictionaries["train"][(target_date, control_date)][fold]["control"] = control_train
            label_dictionaries["train"][(target_date, control_date)][fold][target_disorder] = target_train
            ## Development
            for td in test_date_ranges:
                target_development = dict((x,y) for x,y in test_label_dictionaries[td]["train"][fold]["dev"].items() if y != "control")
                control_development = dict((x,y) for x,y in test_label_dictionaries[td]["train"][fold]["dev"].items() if y == "control")
                label_dictionaries["dev"][td][fold]["control"] = control_development
                label_dictionaries["dev"][td][fold][target_disorder] = target_development
    ## Test
    for td in test_date_ranges:
        target_test = dict((x,y) for x,y in test_label_dictionaries[td]["test"][1]["test"].items() if y != "control")
        control_test = dict((x,y) for x,y in test_label_dictionaries[td]["test"][1]["test"].items() if y == "control")
        label_dictionaries["test"][td][1]["control"] = control_test
        label_dictionaries["test"][td][1][target_disorder] = target_test    
    return label_dictionaries, train_date_combinations, train_date_ranges, test_date_ranges


def create_splits(config,
                  increment_sample_seed=True,
                  train_date_distribution=None,
                  test_date_distribution=None):
    """
    Create train/dev/test splits for temporal effects experiment.

    Args:
        config (dict): Experiment configuration file
        train_date_distribution (str or None): Optional cached date distribution
        test_date_distribution (str or None): Optional cached date distribution
    
    Returns:
        label_dictionaries (dict): Mapping between processed user files and fold/time split
        train_date_combinations (list of tuple): (target, control) date combinations in source domain
        train_date_ranges (list of str): Source domain time periods
        test_date_ranges (list of str): Target domain time periods
        time_boundaries (dict): Mapping between time periods and min/max time boundaries
    """
    ## Identify Datasets
    train_dataset = config["train_data"]
    test_dataset = config["test_data"]
    ## Set Random Seed (for sampling)
    np.random.seed(config["random_seed"])
    ## Case 1: Within-domain
    if train_dataset == test_dataset:
        ## Load Timestamps (or used Cached Version)
        if train_date_distribution is None:
            ## Load Metadata
            metadata = train.load_dataset_metadata(train_dataset,
                                                   config["target_disorder"],
                                                   config["random_seed"])
            ## Get Time Distribution
            filenames, X_time = retrieve_temporal_breakdown(metadata,
                                                            config,
                                                            "train")
        else:
            ## Load Time Distribution
            LOGGER.info(f"Loading Existing Date Distribution: {train_date_distribution}")
            train_date_distribution = joblib.load(train_date_distribution)
            filenames = train_date_distribution["filenames"]
            X_time = train_date_distribution["X_time"]
            config["date_boundaries"]["train"] = train_date_distribution["date_boundaries"]
            config["date_boundaries"]["test"] = train_date_distribution["date_boundaries"]
            ## Load Metadata
            metadata = train.load_dataset_metadata(train_dataset,
                                                   train_date_distribution["target_disorder"],
                                                   train_date_distribution["random_seed"])
        ## Get Label Dictionaries
        train_label_dictionaries, train_time_boundaries = create_temporal_splits(metadata=metadata,
                                                                                 filenames=filenames,
                                                                                 X_time=X_time,
                                                                                 config=config,
                                                                                 data_split="train",
                                                                                 increment_sample_seed=increment_sample_seed)
        test_label_dictionaries = train_label_dictionaries.copy()
        test_time_boundaries = train_time_boundaries.copy()
    ## Case 2: Across Domains
    else:
        ## Source domain
        if train_date_distribution is None:
            ## Load Metadata
            train_metadata = train.load_dataset_metadata(train_dataset,
                                                        config["target_disorder"],
                                                        config["random_seed"])
            ## Load Timestamps
            train_filenames, X_time_train = retrieve_temporal_breakdown(train_metadata,
                                                                        config,
                                                                        "train")
        else:
            ## Load Time Distribution
            LOGGER.info(f"Loading Existing Training Date Distribution: {train_date_distribution}")
            train_date_distribution = joblib.load(train_date_distribution)
            train_filenames = train_date_distribution["filenames"]
            X_time_train = train_date_distribution["X_time"]
            config["date_boundaries"]["train"] = train_date_distribution["date_boundaries"]
            ## Load Metadata
            train_metadata = train.load_dataset_metadata(train_dataset,
                                                         train_date_distribution["target_disorder"],
                                                         train_date_distribution["random_seed"])
        ## Target Domain
        if test_date_distribution is None:
            ## Load Metadata
            test_metadata = train.load_dataset_metadata(test_dataset,
                                                        config["target_disorder"],
                                                        config["random_seed"])
            ## Load Timestamps
            test_filenames, X_time_test = retrieve_temporal_breakdown(test_metadata,
                                                                      config,
                                                                      "test")
        else:
            ## Load Time Distribution
            LOGGER.info(f"Loading Existing Training Date Distribution: {test_date_distribution}")
            test_date_distribution = joblib.load(test_date_distribution)
            test_filenames = test_date_distribution["filenames"]
            X_time_test = test_date_distribution["X_time"]
            config["date_boundaries"]["test"] = test_date_distribution["date_boundaries"]
            ## Load Metadata
            test_metadata = train.load_dataset_metadata(test_dataset,
                                                        test_date_distribution["target_disorder"],
                                                        test_date_distribution["random_seed"])
        ## Get Label Dictionaries
        train_label_dictionaries, train_time_boundaries = create_temporal_splits(metadata=train_metadata,
                                                                                 filenames=train_filenames,
                                                                                 X_time=X_time_train,
                                                                                 config=config,
                                                                                 data_split="train",
                                                                                 increment_sample_seed=increment_sample_seed)
        test_label_dictionaries, test_time_boundaries = create_temporal_splits(metadata=test_metadata,
                                                                               filenames=test_filenames,
                                                                               X_time=X_time_test,
                                                                               config=config,
                                                                               data_split="test",
                                                                               increment_sample_seed=increment_sample_seed)     
    ## Merge Label Dictionaries
    label_dictionaries, train_date_combinations, train_date_ranges, test_date_ranges = merge_dictionaries(train_label_dictionaries,
                                                                                 test_label_dictionaries,
                                                                                 config)
    ## Time Boundary Map (Union)
    time_boundaries = {**train_time_boundaries, **test_time_boundaries}
    return label_dictionaries, train_date_combinations, train_date_ranges, test_date_ranges, time_boundaries

def filter_user_data_by_time(filename,
                             min_date,
                             max_date,
                             data_folder):
    """
    Isolate posts from a processed time period to be within
    a min and max date.

    Args:
        filename (str): Path to processed user data file
        min_date (datetime): Start time boundary
        max_date (datetime): End time boundary
        data_folder (str): Path to temporary folder for storing filtered data
    
    Returns:
        filename (str): Original processed filename
        filtered_filename (str): New filtered data filename
    """
    ## Format File
    min_date_iso = min_date.isoformat()
    max_date_iso = max_date.isoformat()
    base_filename = os.path.basename(filename)
    if ".tweets.tar.gz" in base_filename:
        username = base_filename.split(".tweets.tar.gz")[0]
    elif "posts" in base_filename:
        username = base_filename.split(".posts.tar.gz")[0]
    elif "comments" in base_filename:
        username = base_filename.split(".comments.tar.gz")[0]
    elif "submissions" in base_filename:
        username = base_filename.split(".submissions.tar.gz")[0]
    filtered_filename = "{}{}_{}_{}.tar.gz".format(data_folder, username, min_date_iso, max_date_iso)
    ## Load User Data
    with gzip.open(filename, "r") as the_file:
        user_data = json.load(the_file)
    ## Filter Data
    filtered_data = []
    for record in user_data:
        record_time = datetime.fromtimestamp(record["created_utc"]).date()
        if record_time < min_date or record_time >= max_date:
            continue
        filtered_data.append(record)
    ## Dump
    with gzip.open(filtered_filename, "wt") as the_file:
        json.dump(filtered_data, the_file)
    ## Update Permissions
    _ = os.system(f"chmod 750 {filtered_filename}")
    _ = os.system(f"chown {USERNAME}:ouat {filtered_filename}")
    return (filename, filtered_filename)

def write_temporal_data_splits(config,
                               label_dictionaries,
                               train_date_combinations,
                               train_date_ranges,
                               test_date_ranges,
                               time_boundaries):
    """
    Create copies of processed user data that meets different time boundary filters

    Args:
        config (dict): Experiment configuration
        label_dictionaries (dict): Cross validation/training splits
        train_date_combinations (list of tuple): (target, control) date combinations in source domain
        train_date_ranges (list): List of time periods in source domain
        test_date_ranges (list): List of time periods in target domain
        time_boundaries (dict): Mapping between time period and min/max date
    
    Returns:
        date_aware_label_dictionaries (dict): Same structure as label_dictionaries, but files
                    are the temporally-filtered versions
    """
    ## Create Temporary Folder
    data_folder = "{}data/".format(config["outdir"])
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    ## Update Permissions of Data Folder
    _ = os.system(f"chmod 750 {data_folder}")
    _ = os.system(f"chown {USERNAME}:ouat {data_folder}")
    ## Initialize Updated Label Dictionary
    folds = list(range(1, config["kfolds"]+1))
    date_aware_label_dictionaries = {
                            "train":{c:{f:{} for f in folds} for c in train_date_combinations},
                            "dev":{td:{f:{} for f in folds} for td in test_date_ranges},
                            "test":{td:{1:{}} for td in test_date_ranges},
    }
    ## Cache of Processed File Mappings
    processed_file_map = {}
    ## Initialize Multiprocessing Pool
    mp = Pool(config["jobs"])
    ## Cycle Through Data Splits
    for dsplit, dsplit_dict in label_dictionaries.items():
        for date_index, date_index_dict in dsplit_dict.items():
            for fold, fold_dict in date_index_dict.items():
                for group, group_labels in fold_dict.items():
                    ## Identify Appropriate Date Range
                    if dsplit == "train" and group == config["target_disorder"]:
                        date_range = time_boundaries[date_index[0]]
                    elif dsplit == "train" and group == "control":
                        date_range = time_boundaries[date_index[1]]
                    else:
                        date_range = time_boundaries[date_index]
                    ## Parse Date Range
                    min_date, max_date = date_range
                    min_date = pd.to_datetime(min_date).date()
                    max_date = pd.to_datetime(max_date).date()
                    ## Identify Files That Still Need to Be Processed
                    files_to_process = []
                    for g in sorted(group_labels):
                        if g not in processed_file_map:
                            processed_file_map[g] = {}
                        if date_range not in processed_file_map[g]:
                            processed_file_map[g][date_range] = None
                            files_to_process.append(g)
                    ## Apply Processing as Necessary
                    if len(files_to_process) > 0:
                        ## Processing Helper
                        mp_helper = partial(filter_user_data_by_time,
                                            min_date=min_date,
                                            max_date=max_date,
                                            data_folder=data_folder)
                        ## Process Files
                        res = dict(tqdm(mp.imap_unordered(mp_helper, files_to_process),
                                        total=len(files_to_process),
                                        leave=True,
                                        desc="{} {} {} {}".format(dsplit, fold, date_index, group),
                                        file=sys.stdout))
                        ## Update Filepaths
                        for f_old, f_new in res.items():
                            processed_file_map[f_old][date_range] = f_new
                    ## Update Label Dictionary
                    date_aware_label_dictionaries[dsplit][date_index][fold][group] = {}
                    for f_old, f_lbl in group_labels.items():
                        f_new = processed_file_map[f_old][date_range]
                        date_aware_label_dictionaries[dsplit][date_index][fold][group][f_new] = f_lbl
    ## Close Pool
    mp.close()
    ## Return
    return date_aware_label_dictionaries
    
def _create_temporal_job_script(config_file,
                                target_train_date,
                                control_train_date,
                                split_file,
                                fold,
                                config,
                                memory=64,
                                models_dir=None):
    """
    Format a bash script with particular parameters for one job in the full temporal
    effects experiment.

    Args:
        config_file (str): Path to config file
        target_train_date (str): Source domain target disorder time period
        control_train_date (str): Source domain control group time period
        split_file (str): Path to .joblib file containing date aware splits
        fold (int): Cross validation fold
        config (dict): Experiment configuration
        memory (int): Request for memory allocation (in GB)
        models_dir (str or None)
    
    Returns:
        script (str): Formatted bash script
        exp_name (str): ID for the exepriment
    """
    exp_name = "{}_{}_{}_{}".format(config.get("experiment_name"), target_train_date, control_train_date, fold)
    script = """
    #!/bin/bash
    #$ -cwd
    #$ -S /bin/bash
    #$ -m eas
    #$ -e /home/kharrigian/gridlogs/python/{}.err
    #$ -o /home/kharrigian/gridlogs/python/{}.out
    #$ -pe smp 8
    #$ -l 'gpu=0,mem_free={}g,ram_free={}g'

    ## Move to Home Directory (Place Where Virtual Environments Live)
    cd /home/kharrigian/
    ## Activate Conda Environment
    source .bashrc
    conda activate mental-health
    ## Move To Run Directory
    cd /export/fs03/a08/kharrigian/mental-health/
    ## Run Script
    python ./scripts/experiment/temporal_effects/temporal_effects_helper.py --config {} --target_date {} --control_date {} --splits {} --fold {} {}
    """.format(
        exp_name,
        exp_name,
        memory,
        memory,
        config_file,
        target_train_date,
        control_train_date,
        split_file,
        fold,
        "--models_dir {}".format(models_dir) if models_dir is not None else "").rstrip()
    return script, exp_name


def get_running_jobs():
    """
    Identify all jobs running for the user specified in this script's configuration

    Args:
        None

    Returns:
        running_jobs (list): List of integer job IDs on the CLSP grid
    """
    jobs_res = subprocess.check_output(f"qstat -u {USERNAME}", shell=True)
    jobs_res = jobs_res.decode("utf-8").split("\n")[2:-1]
    running_jobs = [int(i.split()[0]) for i in jobs_res]
    return running_jobs

def hold_for_complete_jobs(scheduled_jobs,
                           wait_time=20):
    """
    Sleep until all jobs scheduled have been completed

    Args:
        scheduled_jobs (list): Output of schedule_jobs
        wait_time (int): How long (in seconds) to wait between checking
                         for job completions
    
    Returns:
        None
    """
    ## Sleep Unitl Jobs Complete
    complete_jobs = []
    sleep_count = 0
    while sorted(complete_jobs) != sorted([s[0] for s in scheduled_jobs]):
        ## Get Running Jobs
        running_jobs = get_running_jobs()
        ## Look for Newly Completed Jobs
        newly_completed_jobs = []
        for s in scheduled_jobs:
            if s[0] not in running_jobs and s[0] not in complete_jobs:
                newly_completed_jobs.append(s[0])
        ## Sleep If No Updates, Otherwise Update Completed Job List and Reset Counter
        if len(newly_completed_jobs) == 0:
            if sleep_count % 5 == 0:
                LOGGER.info("Temporal Effects jobs still running. Continuing to sleep.")
            sleep(wait_time)
            sleep_count += 1
        else:
            complete_jobs.extend(newly_completed_jobs)
            LOGGER.info("Newly finished jobs: {}".format(newly_completed_jobs))
            n_jobs_complete = len(complete_jobs)
            n_jobs_remaining = len(scheduled_jobs) - n_jobs_complete
            LOGGER.info(f"{n_jobs_complete} jobs complete. {n_jobs_remaining} jobs remaining.")
            sleep_count = 0


def run_temporal_cross_validation(config,
                                  date_aware_label_dictionaries,
                                  train_date_combinations,
                                  train_date_ranges,
                                  test_date_ranges,
                                  time_boundaries,
                                  models_dir=None):
    """
    Schedule component jobs as part of the temporal cross-validation procedure

    Args:
        config (dict): Experiment configuration
        date_aware_label_dictionaries (dict): Cross validation splits (temporally aware)
        train_date_combinations (list of tuple): (target, control) time period combos in source domain
        train_date_ranges (list of str): Time periods in source domain
        test_date_ranges (list of str): Time periods in target domain
        time_boundaries (dict): Map beween time periods and min/max data
        models_dir (str or None): Path to cached model directory for multiple experiments
    
    Returns:
        None, writes data to disk
    """
    ## Results Directory
    cv_outdir = "{}cross_validation/".format(config.get("outdir"))
    if not os.path.exists(cv_outdir):
        os.makedirs(cv_outdir)
    ## Cache Splits
    split_file = f"{cv_outdir}splits.joblib"
    _ = joblib.dump(date_aware_label_dictionaries, split_file)
    ## Create Temp Directory
    temp_job_dir = f"{cv_outdir}temp/"
    if not os.path.exists(temp_job_dir):
        os.makedirs(temp_job_dir)
    ## Cache For Job Data
    scheduled_jobs = []
    ## Cycle Through Data Splits to Create Jobs
    if config["mixed_time_windows"]:
        train_date_groups = list(product(train_date_ranges, train_date_ranges))
    else:
        train_date_groups = [(t, t) for t in train_date_ranges]
    folds = list(range(1,config["kfolds"]+1))
    config_file = config.get("outdir") + "config.json"
    for target_train_date, control_train_date in train_date_groups:
        for fold in folds:
            ## Format Script
            job_script, job_name = _create_temporal_job_script(config_file,
                                                               target_train_date,
                                                               control_train_date,
                                                               split_file,
                                                               fold,
                                                               config,
                                                               models_dir=models_dir)
            ## Write Bash File
            job_file = f"{temp_job_dir}{job_name}.sh"
            with open(job_file, "w") as the_file:
                the_file.write("\n".join([i.lstrip() for i in job_script.split("\n")]))
            ## Start Job
            qsub_call = f"qsub {job_file}"
            job_id = subprocess.check_output(qsub_call, shell=True)
            job_id = int(job_id.split()[2])
            ## Cache Job Data
            scheduled_jobs.append((job_id, job_name))
    ## Wait For Jobs to Finish
    _ = hold_for_complete_jobs(scheduled_jobs)
    ## Remove Temp Directory
    _ = os.system(f"rm -rf {temp_job_dir}")

def concatenate_cross_validation_predictions(config,
                                             date_aware_label_dictionaries,
                                             time_boundaries):
    """
    Concatenate cross validation predictions into a single DataFrame

    Args:
        config (dict): Experiment configuration
        date_aware_label_dictionaries (dict): Date aware ground truth splits
        time_boundaries (dict): Map between time periods and min/max dates
    
    Returns:
        predictions (pandas DataFrame): Raw probability/class predictions and ground truth
    """
    ## Identify Results Directories
    cv_dir = "{}cross_validation/".format(config.get("outdir"))
    cv_res_dirs = [i for i in glob(cv_dir + "*") if os.path.isdir(i)]
    ## Process Results Independently
    predictions = []
    for cv_res in cv_res_dirs:
        ## Parse Metadata
        exp_name = os.path.basename(cv_res)
        target_train_date, control_train_date, fold = exp_name.split("_")
        fold = int(fold)
        ## Check for and Load Predictions
        pred_file = f"{cv_res}/predictions.json.gz"
        if not os.path.exists(pred_file):
            LOGGER.info(f"WARNING: Did not find expected predictions file '{pred_file}'")
            continue
        with gzip.open(f"{cv_res}/predictions.json.gz", "r") as the_file:
            res_preds = json.load(the_file)[exp_name]
        ## Parse Training Predictions
        train_gt_labels = date_aware_label_dictionaries["train"][(target_train_date, control_train_date)][fold]
        train_preds = pd.Series(res_preds["train"]).reset_index().rename(columns={"index":"source",0:"y_pred"})
        train_preds["y_true"] = train_preds["source"].map(lambda i: 0 if i in train_gt_labels["control"] else 1)
        train_preds["target_train_min_date"] = time_boundaries[target_train_date][0]    
        train_preds["target_train_max_date"] = time_boundaries[target_train_date][1]
        train_preds["control_train_min_date"] = time_boundaries[control_train_date][0]
        train_preds["control_train_max_date"] = time_boundaries[control_train_date][1]
        train_preds["user_min_date"] = train_preds["y_true"].map(lambda i: time_boundaries[target_train_date][0] if i == 0 else 
                                                                           time_boundaries[control_train_date][0])
        train_preds["user_max_date"] = train_preds["y_true"].map(lambda i: time_boundaries[target_train_date][1] if i == 0 else 
                                                                           time_boundaries[control_train_date][1])    
        train_preds["group"] = "train"
        train_preds["fold"] = fold
        train_preds["user_id_str"] = train_preds["source"].map(os.path.basename).map(lambda i: i[:-29])
        train_preds["user_in_training"] = True
        predictions.append(train_preds)
        ## Parse Development Predictions
        for test_date, test_date_dict in res_preds["dev"].items():
            test_gt_labels = date_aware_label_dictionaries["dev"][test_date][fold]
            test_preds = pd.Series(test_date_dict).reset_index().rename(columns={"index":"source",0:"y_pred"})
            test_preds["y_true"] = test_preds["source"].map(lambda i: 0 if i in test_gt_labels["control"] else 1)
            test_preds["target_train_min_date"] = time_boundaries[target_train_date][0]    
            test_preds["target_train_max_date"] = time_boundaries[target_train_date][1]
            test_preds["control_train_min_date"] = time_boundaries[control_train_date][0]
            test_preds["control_train_max_date"] = time_boundaries[control_train_date][1]
            test_preds["user_min_date"] = test_preds["y_true"].map(lambda i: time_boundaries[test_date][0])
            test_preds["user_max_date"] = test_preds["y_true"].map(lambda i: time_boundaries[test_date][1])
            test_preds["group"] = "dev"
            test_preds["fold"] = fold
            test_preds["user_id_str"] = test_preds["source"].map(os.path.basename).map(lambda i: i[:-29])
            test_preds["user_in_training"] = test_preds.user_id_str.isin(train_preds.user_id_str)
            predictions.append(test_preds)
    ## Concatenate
    predictions = pd.concat(predictions).reset_index(drop=True).copy()
    ## Cache Predictions
    predictions.to_csv(f"{cv_dir}predictions.csv", index=False)
    ## Remove Intermediate Directories
    for cv_dir in cv_res_dirs:
        _ = os.system(f"rm -rf {cv_dir}")
    return predictions

def get_split_sizes(label_dictionaries):
    """
    Examine number of users in each split of a label dictionary

    Args:
        label_dictionaries (dict): Cross validation splits
    
    Returns:
        size_pivot (pandas DataFrame): Summary of splits over time/fold
    """
    sizes = []
    for dsplit, dsplit_dict in label_dictionaries.items():
        for train_date, train_date_dict in dsplit_dict.items():
            for fold, fold_dict in train_date_dict.items():
                for lbl, lbl_dict in fold_dict.items():
                    n = len(lbl_dict)
                    sizes.append([dsplit, train_date, fold, lbl, n])
    sizes = pd.DataFrame(sizes,
                        columns=["group","date","fold","label","size"])
    size_pivot = pd.pivot_table(sizes,
                                index=["group","date","label"],
                                columns="fold",
                                values="size",
                                aggfunc=max).fillna(0).astype(int)
    return size_pivot

def get_prediction_combo_sizes(predictions,
                               mixed_time_windows=True):
    """
    Get summary of data set sizes in predictions

    Args:
        predictions (pandas DataFrame): Raw predictions
    
    Returns:
        size_pivot (pandas DataFrame): Summary of data set sizes per split
    """
    ## Date Combinations
    train_date_combos = predictions[["target_train_min_date","control_train_min_date"]].drop_duplicates().values
    if mixed_time_windows:
        test_date_combos = list(product(predictions["user_min_date"].unique(), predictions["user_min_date"].unique()))
    else:
        test_date_combos = [(t, t) for t in predictions["user_min_date"].unique()]
    ## Cross Validation Folds
    folds = sorted(predictions["fold"].unique())
    ## Cycle Through Combinations, Recording Sizes
    sizes = []
    for group in ["train","dev"]:
        for train_dc in train_date_combos:
            for test_dc in test_date_combos:
                for fold in folds:
                    pred_subset = predictions.loc[(predictions["group"]==group)&
                                                (predictions["fold"]==fold)&
                                                (predictions["target_train_min_date"]==train_dc[0])&
                                                (predictions["control_train_min_date"]==train_dc[1])]
                    for lbl_name, lbl_val, lbl_ind in zip(["control","depression"],
                                                          [0, 1],
                                                          [1, 0]):
                        lbl_pred_subset = pred_subset.loc[(pred_subset["user_min_date"]==test_dc[lbl_ind])&
                                                          (pred_subset["y_true"]==lbl_val)]
                        n = len(lbl_pred_subset)
                        n_unseen = (~lbl_pred_subset["user_in_training"]).sum()
                        sizes.append([
                            group,
                            train_dc[0],
                            train_dc[1],
                            test_dc[0],
                            test_dc[1],
                            fold,
                            lbl_name,
                            n,
                            n_unseen,
                        ])
    ## Format Size
    sizes = pd.DataFrame(sizes,
                        columns=["group",
                                "target_train_date",
                                "control_train_date",
                                "target_test_date",
                                "control_test_date",
                                "fold",
                                "label",
                                "n",
                                "n_unseen"])
    ## Create Pivot Table
    size_pivot = pd.pivot_table(sizes,
                                index=["group","target_train_date","control_train_date","label"],
                                columns=["fold","target_test_date","control_test_date"],
                                values=["n","n_unseen"],
                                aggfunc=max)
    return size_pivot

def get_scores(y_true,
               y_pred_score,
               threshold=0.5):
    """
    Score predictions

    Args:
        y_true (array): Ground truth, binary labels
        y_pred_score (array): Predicted positive class probabilities
        threshold (float): Positive class probability threshold
    
    Returns:
        score_dict (dict): Scores for various ML metrics
    """
    fpr, tpr, thres = metrics.roc_curve(y_true, y_pred_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    y_pred = (y_pred_score > threshold).astype(int)
    prec = metrics.precision_score(y_true, y_pred, average="binary")
    recall = metrics.recall_score(y_true, y_pred, average="binary")
    f1 = metrics.f1_score(y_true, y_pred, average="binary")
    accuracy = metrics.accuracy_score(y_true, y_pred)
    score_dict = {
        "fpr":list(fpr),
        "tpr":list(tpr),
        "thresholds":thres,
        "auc":auc,
        "precision":prec,
        "recall":recall,
        "f1":f1,
        "accuracy":accuracy
    }
    return score_dict

def score_predictions(config,
                      predictions,
                      threshold=0.5):
    """
    Evaluate performance for various source/target combinations

    Args:
        config (dict): Experiment configuration parameters
        predictions (pandas DataFrame): Raw cross-validation predictions
        threshold (float): Positive class probability threshold
    
    Returns:
        scores (pandas DataFrame): Scores per source/target/time/fold split
    """
    ## Date Combinations
    train_date_combos = predictions[["target_train_min_date","control_train_min_date"]].drop_duplicates().values
    if config["mixed_time_windows"]:
        test_date_combos = list(product(predictions["user_min_date"].unique(), predictions["user_min_date"].unique()))
    else:
        test_date_combos = [(t, t) for t in predictions["user_min_date"].unique()]
    ## Cross Validation Folds
    folds = sorted(predictions["fold"].unique())
    ## Cycle Through Combinations
    scores = []
    for group in ["train","dev"]:
        for train_dc in train_date_combos:
            for test_dc in test_date_combos:
                for fold in folds:
                    pred_subset = predictions.loc[(predictions["group"]==group)&
                                                  (predictions["fold"]==fold)&
                                                  (predictions["target_train_min_date"]==train_dc[0])&
                                                  (predictions["control_train_min_date"]==train_dc[1])]
                    control_pred_subset = pred_subset.loc[(pred_subset["user_min_date"]==test_dc[1])&
                                                          (pred_subset["y_true"]==0)]
                    target_pred_subset = pred_subset.loc[(pred_subset["user_min_date"]==test_dc[0])&
                                                          (pred_subset["y_true"]==1)]
                    if len(control_pred_subset) == 0 or len(target_pred_subset) == 0:
                        continue
                    for user_seen_set, user_seen_name in zip([[False,True],[False],[True]],
                                                             ["overall","unseen","seen"]):
                        control_pred_seen_sub = control_pred_subset.loc[control_pred_subset["user_in_training"].isin(user_seen_set)]
                        target_pred_seen_sub = target_pred_subset.loc[target_pred_subset["user_in_training"].isin(user_seen_set)]
                        if len(control_pred_seen_sub) == 0 or len(target_pred_seen_sub) == 0:
                            continue
                        combined_pred_sub = pd.concat([control_pred_seen_sub, target_pred_seen_sub])
                        combined_pred_scores = get_scores(combined_pred_sub["y_true"].values,
                                                          combined_pred_sub["y_pred"].values,
                                                          threshold)
                        combined_pred_scores["group"] = group
                        combined_pred_scores["fold"] = fold
                        combined_pred_scores["seen_subset"] = user_seen_name
                        combined_pred_scores["n_control"] = len(control_pred_seen_sub)
                        combined_pred_scores["n_target"] = len(target_pred_seen_sub)
                        combined_pred_scores["support"] = len(combined_pred_sub)
                        combined_pred_scores["target_train"] = train_dc[0]
                        combined_pred_scores["control_train"] = train_dc[1]
                        combined_pred_scores["target_test"] = test_dc[0]
                        combined_pred_scores["control_test"] = test_dc[1]
                        scores.append(combined_pred_scores)
    ## Format Scores
    scores = pd.DataFrame(scores)
    ## Cache
    scores.to_csv("{}cross_validation/scores.csv".format(config.get("outdir")), index=False)
    return scores

def score_heatmap(scores,
                  metric="f1",
                  group="dev",
                  seen_subset="overall"):
    """
    Create a heatmap of scores, comparing performance
    under different source to target date combinations.

    Args:
        scores (pandas DataFrame): Summary statistic dataframe
        metric (str): Statistic to plot
        group (str): Which split to split (e.g. "train" vs. "dev")
        seen_subset (str): How to filter seen users
    
    Returns:
        fig, ax (tuple): Matplotlib objects
        score_pivot (pandas DataFrame): Mean and Standard Deviation
    """
    ## Filter
    score_subset = scores.loc[scores["seen_subset"]==seen_subset]
    ## Aggregate Score
    score_pivot = pd.pivot_table(score_subset,
                                 index=["group","target_train","control_train"],
                                 columns=["target_test","control_test"],
                                 values=metric,
                                 aggfunc=[np.nanmean, np.nanstd])
    ## Group Subset
    score_pivot = score_pivot.loc[group]
    score_pivot = score_pivot.iloc[::-1]
    ## Identify Dates
    train_dates = score_pivot["nanmean"].columns.tolist()
    test_dates = score_pivot["nanmean"].index.tolist()
    ## Create Figure
    fig, ax = plt.subplots(1, 1, figsize=(10,5.8), sharey=True)
    m = ax.imshow(score_pivot["nanmean"].values,
                  cmap=plt.cm.Purples,
                  aspect="auto",
                  interpolation="nearest",
                  alpha=.75)
    ax.set_xticks(list(range(score_pivot["nanmean"].shape[1])))
    ax.set_xticklabels(["Target: {}\nControl: {}".format(t[0],t[1]) for t in train_dates], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(list(range(score_pivot["nanmean"].shape[0])))
    ax.set_yticklabels(["Target: {}\nControl: {}".format(t[0],t[1]) for t in test_dates], fontsize=6)
    ax.set_ylim(-.5, score_pivot["nanmean"].shape[0]-.5)
    for i, row in enumerate(score_pivot["nanmean"].values):
        for j, val in enumerate(row):
            try:
                if pd.isnull(val):
                    ax.text(j, i, "-", fontsize=5, ha="center", va="center")
                else:
                    ax.text(j,
                            i,
                            "{:.2f} ({:.2f})".format(val, score_pivot["nanstd"].values[i,j]),
                            ha="center",
                            va="center",
                            fontsize=5)
            except IndexError:
                continue
    ax.set_xlabel("Evaluation Times")
    ax.set_ylabel("Training Times")
    ax.set_title(f"{seen_subset.title()} - {group.title()} - {metric.title()}", loc="left", fontweight="bold")
    fig.tight_layout()
    return (fig, ax), score_pivot

def plot_hypothesis_1(scores,
                      group="dev",
                      metric="f1",
                      seen_subset="unseen"):
    """
    H1: Classification is easier when there is a temporal mismatch 
    between classes (e.g. depression, control group)
        - F(x, x, x’, x’) < F(x, y, x’, y’),
        - F(y, y, y’, y’) < F(x, y, x’, y’)
        - F(x, x, x’, x’) <  F(y, x, y’, x’)
        - F(y, y, y’, y’) < F(y, x, y’, x’)
            - Why? Because when there is an additional temporal element that separates 
            the control and target groups that should in theory make classification easier
    
    Args:
        scores (pandas DataFrame)
    
    Returns:
        fig, ax (matplotlib objects)
    """
    ## Get Subset
    score_subset = scores.loc[(scores["group"]==group)&
                              (scores["seen_subset"]==seen_subset)].copy()
    ## Get Hypothesis Subsets
    scores_same = score_subset.loc[(score_subset["target_train"]==score_subset["control_train"])&
                                   (score_subset["target_train"]==score_subset["target_test"])&
                                   (score_subset["target_test"]==score_subset["control_test"])]
    scores_mismatch = score_subset.loc[(score_subset["target_train"]!=score_subset["control_train"])&
                                       (score_subset["target_test"]!=score_subset["control_test"])&
                                       (score_subset[["target_train","control_train"]].apply(tuple,axis=1)==\
                                        score_subset[["target_test","control_test"]].apply(tuple,axis=1))]
    ## Temporal Groups
    same_scores_dates = sorted(scores_same[["target_train","control_train"]].drop_duplicates().values, key=lambda x:x[0])
    mismatch_scores_dates = scores_mismatch[["target_train","control_train"]].drop_duplicates().\
                            sort_values(["target_train","control_train"]).values
    ## Create Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    xticks = []
    xticklabels = []
    min_height, max_height = 1, 0
    for h, (hgroup, hdf) in enumerate(zip([same_scores_dates, mismatch_scores_dates], [scores_same, scores_mismatch])):
        bar_width = 1/len(hgroup)
        hgroup_scores = []
        for b, (tar_date, con_date) in enumerate(hgroup):
            plot_data = hdf.loc[(hdf["target_train"]==tar_date)&(hdf["control_train"]==con_date)]
            if plot_data[metric].min() < min_height:
                min_height = plot_data[metric].min()
            if plot_data[metric].max() > max_height:
                max_height = plot_data[metric].max()
            mean, std = plot_data[metric].mean(), plot_data[metric].std()
            standard_error = std / np.sqrt(len(plot_data) - 1)
            hgroup_scores.extend(plot_data[metric].tolist())
            ax.bar(h+b*bar_width,
                   mean,
                   yerr = standard_error,
                   color = f"C{h}",
                   alpha = .5,
                   align = "center",
                   width = bar_width*.95,
                   label = {0:"Aligned Time",1:"Time Mismatch"}[h] if b == 0 else "")
            ax.scatter([h + b*bar_width] * len(plot_data),
                       plot_data[metric].tolist(),
                       color = f"C{h}",
                       zorder = 10)
            xticks.append(h + b*bar_width)
            xticklabels.append(f"{tar_date} : {con_date}")
        mean_hgroup = np.mean(hgroup_scores)
        std_hgroup = np.std(hgroup_scores)
        standard_error = std_hgroup / np.sqrt(len(hgroup_scores)-1)
        ax.bar(2 + (h+1)*0.4,
               mean_hgroup,
               yerr = standard_error,
               color = f"C{h}",
               alpha = .5,
               width = 0.4*.95)
        xticks.append(2 + (h+1)*0.4)
        xticklabels.append("Avg. Aligned" if h == 0 else "Avg. Mismatch")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=6)
    ax.set_ylim(min_height * .99, max_height * 1.01)
    ax.set_ylabel(f"{metric.title()} Score", fontweight="bold")
    ax.set_xlabel("Disorder : Control Time Period", fontweight="bold")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig, ax

def plot_hypothesis_2(scores,
                      group="dev",
                      metric="f1",
                      seen_subset="unseen"):
    """
    H2: Classification performance worsens when applying a model trained on data
    entirely from one time period to another
        - F(x, x, y’, y’) < F(y, y, y’, y’)
        - F(y, y, x’, x’) < F(x, x, x’, x’)
            - Why? Because if temporal element is strong, we would see degradation
            in performance when you take models trained on old data and apply it to 
            newer data (or vice versa). This would signal there is a change in language
            usage over time and the signal relevant to each class is altered
    
    H2.5: Classification performance worsens more dramatically as the delta in time grows
        - F(x, x, x’, x’) > F(x, x, (x+1)’, (x+1)’) > F(x, x, (x+2)’, (x+2)’) > … > F(x, x, (x+m)’, (x+m)’)
        - F(x, x, x’, x’) > F(x, x, (x-1)’, (x-1)’) > F(x, x, (x-2)’, (x-2)’) > … > F(x, x, (x-m)’, (x-m)’)
            - Why? Change in language should become more severe over time and cause continual degradation of performance
    """
    ## Get Subset
    score_subset = scores.loc[(scores["group"]==group)&
                              (scores["seen_subset"]==seen_subset)].copy()
    ## Identify Test Years
    test_years = score_subset.loc[score_subset["target_test"]==score_subset["control_test"]][["target_test","control_test"]].drop_duplicates().values
    test_years = sorted(test_years, key=lambda x: x[0])
    ## Create Figure
    fig, ax = plt.subplots(figsize=(10,5.8))
    xticks = []
    xticklabels = []
    for t, (tar_year, con_year) in enumerate(test_years):
        plot_subset = score_subset.loc[(score_subset["target_test"]==tar_year)&
                                       (score_subset["control_test"]==con_year)&
                                       (score_subset["target_train"]==score_subset["control_train"])]
        train_years = sorted(plot_subset[["target_train","control_train"]].drop_duplicates().values,
                             key=lambda x: x[0])
        bar_width = .95 / len(train_years)
        fold_connections = []
        for a, (tar_train, con_train) in enumerate(train_years):
            a_subset = plot_subset.loc[(plot_subset["target_train"]==tar_train)&
                                       (plot_subset["control_train"]==con_train)].sort_values("fold")
            mean, std = a_subset[metric].mean(), a_subset[metric].std()
            standard_error = std / np.sqrt(len(a_subset) - 1)
            ax.bar(t+a*bar_width,
                   mean,
                   yerr = standard_error,
                   color = f"C{t}",
                   alpha = .5,
                   align = "center",
                   width = bar_width * .95,
                   label = tar_year if tar_train == tar_year else "",
                   edgecolor = "black" if tar_train == tar_year else None,
                   linewidth = 2)
            ax.scatter([t + a*bar_width] * len(a_subset),
                       a_subset[metric].tolist(),
                       color = f"C{t}",
                       zorder = 10,
                       alpha=.5)
            xticks.append(t+a*bar_width)
            xticklabels.append(tar_train)
            fold_connections.extend(list(zip([t + a*bar_width] * len(a_subset), a_subset[metric].tolist())))
        for fold in range(len(a_subset)):
            ax.plot([x for i, (x, y) in enumerate(fold_connections) if i % len(a_subset) == fold],
                    [y for i, (x, y) in enumerate(fold_connections) if i % len(a_subset) == fold],
                    color = f"C{t}",
                    alpha = 0.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(f"{metric.title()} Score", fontweight="bold")
    ax.set_xlabel("Training Period", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), frameon=True, title="Test Period")
    fig.tight_layout()
    return fig, ax


def visualize_scores(config,
                     scores):
    """
    Create visualizations for cross validation results

    Args:
        config (dict): Experiment configuration
        scores (pandas DataFrame): Summary statistics
    
    Returns:
        None
    """
    for metric in ["auc","precision","recall","f1","accuracy"]:
        ## Visualize Regular Score Across All Splits
        (fig, ax), _ = score_heatmap(scores,
                                     metric=metric,
                                     seen_subset="unseen")
        fig.savefig("{}cross_validation/unseen_{}.png".format(config.get("outdir"), metric), dpi=150)
        plt.close()
        ## Hypothesis Testing
        if config["train_data"] == config["test_data"]:
            ## Hypothesis 1
            if config["mixed_time_windows"]:
                fig, ax = plot_hypothesis_1(scores,
                                            metric=metric,
                                            group="dev",
                                            seen_subset="unseen")
                fig.savefig("{}cross_validation/hypothesis_1_unseen_{}.png".format(config.get("outdir"), metric), dpi=150)
                plt.close()
            ## Hypothesis 2
            fig, ax = plot_hypothesis_2(scores,
                                        metric=metric,
                                        group="dev",
                                        seen_subset="unseen")
            fig.savefig("{}cross_validation/hypothesis_2_unseen_{}.png".format(config.get("outdir"), metric), dpi=150)
            plt.close()

def main():
    """
    Main procedure. Run cross validation on temporal
    splits and evaluate results

    Args:
        None

    Returns:
        None
    """
    ## Parse command-line arguments
    args = parse_arguments()
    ## Load Configuration, Save Cached Version for Reference
    LOGGER.info("Loading Experiment Configuration")
    config = train.load_configuration(args,
                                      RESULTS_DIR,
                                      "temporal_effects")
    ## Create Temporal Splits
    LOGGER.info("Creating Temporal Data Splits")
    label_dictionaries, train_date_combinations, train_date_ranges, test_date_ranges, time_boundaries = create_splits(config,
                                                                                                                      increment_sample_seed=True,
                                                                                                                      train_date_distribution=args.train_date_distribution,
                                                                                                                      test_date_distribution=args.test_date_distribution)
    ## Write Temporally-Aligned Data to Disk
    LOGGER.info("Writing Temporally-Aligned Data Files")
    date_aware_label_dictionaries = write_temporal_data_splits(config,
                                                               label_dictionaries,
                                                               train_date_combinations,
                                                               train_date_ranges,
                                                               test_date_ranges,
                                                               time_boundaries)
    ## Run Temporal Cross Validation
    LOGGER.info("Scheduling Cross Validation Jobs")
    _ = run_temporal_cross_validation(config,
                                      date_aware_label_dictionaries,
                                      train_date_combinations,
                                      train_date_ranges,
                                      test_date_ranges,
                                      time_boundaries,
                                      models_dir=args.models_dir)
    ## Combine Predictions and Ground Truth into Single Object
    LOGGER.info("Concatenating Cross Validation Results")
    predictions = concatenate_cross_validation_predictions(config,
                                                           date_aware_label_dictionaries,
                                                           time_boundaries)
    ## Identify Relevant Data Set Sizes
    data_split_sizes = get_split_sizes(date_aware_label_dictionaries)
    prediction_sizes = get_prediction_combo_sizes(predictions,
                                                  config["mixed_time_windows"])
    data_split_sizes.to_csv("{}cross_validation/split_sizes.csv".format(config.get("outdir")))
    prediction_sizes.to_csv("{}cross_validation/prediction_sizes.csv".format(config.get("outdir")))
    ## Score Cross Validation Results
    LOGGER.info("Beginning Scoring of Cross Validation Results")
    scores = score_predictions(config,
                               predictions,
                               threshold=0.5)
    ## Visualize Scores
    LOGGER.info("Visualizing Results")
    _ = visualize_scores(config,
                         scores)
    ## Remove Temporal Data splits
    LOGGER.info("Removing Temporal Data Splits")
    _ = os.system("rm -rf {}data/".format(config.get("outdir")))
    ## Procedure Complete
    LOGGER.info("Experiment Complete!")
    
#######################
### Execution
#######################

if __name__ == "__main__":
    _ = main()