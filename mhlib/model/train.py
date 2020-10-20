
#################
### Imports
#################

## Standard Libary
import os
import json
import gzip
from copy import deepcopy
from datetime import datetime

## External Libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import (KFold,
                                     StratifiedKFold)

## Local Modules
from .model import MentalHealthClassifier
from ..util.logging import initialize_logger

#################
### Globals
#################

## Initialize Logger
LOGGER = initialize_logger()

## Key Directories
ROOT_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../") + "/"
DATA_DIR = f"{ROOT_DIR}data/"
MODELS_DIR = f"{ROOT_DIR}models/"

## Processed Data Paths
DATA_PATHS = {
    ("wolohan",):{
                "path":f"{DATA_DIR}processed/reddit/wolohan/",
                "suffix":"comments.tar.gz",
                "meta_suffix":"comments.meta.tar.gz"
              },
    ("clpsych","clpsych_deduped","multitask","merged"):{
                "path":f"{DATA_DIR}processed/twitter/qntfy/",
                "suffix":"tweets.tar.gz",
                "meta_suffix":"tweets.meta.tar.gz"
                },
    ("rsdd", ):{
                "path":f"{DATA_DIR}processed/reddit/rsdd/",
                "suffix":"posts.tar.gz",
                "meta_suffix":"posts.meta.tar.gz"
              },
    ("smhd", ):{
                "path":f"{DATA_DIR}processed/reddit/smhd/",
                "suffix":"posts.tar.gz",
                "meta_suffix":"posts.meta.tar.gz"
              },
    }

## Score Functions
SCORE_FUNCTIONS = {
    "f1":metrics.f1_score,
    "precision":metrics.precision_score,
    "recall":metrics.recall_score,
    "accuracy":metrics.accuracy_score
}

#################
### Functions
#################

def _cache_config_and_setup_dirs(config,
                                 base_outdir):
    """
    Cache a copy of the experiment configuration file
    and setup directories for experiment results.

    Args:
        config (dict): Experiment configuration
        base_outdir (str): Base output directory (e.g. data/results/domain_transfer/ or models/)
    
    Returns:
        exp_outdir (str): Path to output directory for the experiment
    """
    ## Output Filename
    runtime = datetime.now().strftime("%Y%m%d%H%M%S")
    exp = config["experiment_name"]
    outdir = f"{base_outdir}{runtime}-{exp}/"
    ## Replace Grid Search Config If Appicable
    if config.get("grid_search"):
        gs_config_file = config["grid_search_kwargs"]["config"]
        if gs_config_file is None:
            gs_config_file = f"{ROOT_DIR}configurations/hyperparameter_search/_default.json"
        if not os.path.exists(gs_config_file):
            raise FileNotFoundError("Could not find grid search config {}".format(gs_config_file))
        with open(gs_config_file, "r") as the_file:
            config["grid_search_specification"] = json.load(the_file)
    ## Setup Output Directory
    _ = os.makedirs(outdir)
    ## Write Config Copy
    config = deepcopy(config)
    config["outdir"] = outdir
    with open(f"{outdir}config.json", "w") as the_file:
        json.dump(config.copy(), the_file)
    return outdir

def _validate_config_params(config,
                            configuration_type):
    """
    Check that expected configration parameters
    are present and valid.

    Args:
        config (dict): Dictionary of experiment parameters
        configuration_type (str): Either "model","domain_transfer", or "learning_curve"
    
    Returns:
        None
    
    Raises:
        Exceptions if any of the standard configuration parameters are 
        invalid/missing.
    """
    ## Key Check
    primary_keys = ["experiment_name",
                    "train_data",
                    "target_disorder",
                    "random_seed",
                    "kfolds",
                    "test_size",
                    "downsample",
                    "downsample_size",
                    "rebalance",
                    "class_ratio",
                    "model",
                    "model_kwargs",
                    "vocab_kwargs",
                    "grid_search",
                    "grid_search_kwargs",
                    "preprocessing_kwargs",
                    "feature_selector",
                    "feature_selection_kwargs"]
    if configuration_type == "domain_transfer":
        primary_keys.extend(["test_data",
                             "run_on_test",
                             "date_boundaries"])
    elif configuration_type == "temporal_effects":
        primary_keys.extend(["test_data",
                             "date_boundaries",
                             "mixed_time_windows",
                             "balance_date_splits",
                             "run_on_test",
                             "min_users_per_split",
                             "min_posts"])
    preprocessing_keys = [
            "feature_flags",
            "feature_kwargs",
            "standardize",
    ]
    vocab_keys = [
            "filter_negate",
            "filter_upper",
            "filter_punctuation",
            "filter_numeric",
            "filter_user_mentions",
            "filter_url",
            "filter_retweet",
            "filter_stopwords",
            "keep_pronouns",
            "preserve_case",
            "emoji_handling",
            "filter_mh_subreddits",
            "filter_mh_terms",
            "max_tokens_per_document",
            "max_documents_per_user",
            "max_vocab_size",
            "min_token_freq",
            "max_token_freq",
            "ngrams",
            "binarize_counter",
    ]
    grid_search_keys = [
        "config",
        "test_size",
        "dev_frac",
        "dev_in_vocab",
        "score_func",
        "use_multiprocessing"
    ]
    for k in primary_keys:
        if k not in config:
            raise KeyError(f"{k} not found in config file") 
    for k in preprocessing_keys:
        if k not in config.get("preprocessing_kwargs"):
            raise KeyError(f"{k} not found in config preprocessing params") 
    for k in vocab_keys:
        if k not in config.get("vocab_kwargs"):
            raise KeyError(f"{k} not found in config vocab params") 
    for k in grid_search_keys:
        if k not in config.get("grid_search_kwargs"):
            raise KeyError(f"{k} not found in config grid search params")
    ## Check Dataset
    ds_choices = set(["wolohan",
                      "clpsych",
                      "clpsych_deduped",
                      "multitask",
                      "merged",
                      "rsdd",
                      "smhd",
                      ])
    if configuration_type in ["model","domain_transfer"]:
        if config["train_data"] not in ds_choices:
            raise ValueError(f"Configuration parameter 'train_data' incorrectly specified. Must be one of {ds_choices}")
    if configuration_type in ["domain_transfer"]:
        if config["test_data"] not in ds_choices:
            raise ValueError(f"Configuration parameter 'test_data' incorrectly specified. Must be one of {ds_choices}")
    ## Check Primary Arguments
    assert isinstance(config["random_seed"], int)
    assert isinstance(config["kfolds"], int)
    assert config["test_size"] < 1 and config["test_size"] >= 0
    ## Check Vocabulary Arguments
    for k in vocab_keys:
        if k.startswith("filter_") and "mh" not in k:
            assert isinstance(config["vocab_kwargs"][k], bool)
        elif k.startswith("filter_") and "mh" in k:
            assert isinstance(config["vocab_kwargs"][k], str) or config["vocab_kwargs"][k] is None
    assert isinstance(config["vocab_kwargs"]["max_vocab_size"], int) or config["vocab_kwargs"]["max_vocab_size"] is None
    assert isinstance(config["vocab_kwargs"]["min_token_freq"], int)
    assert isinstance(config["vocab_kwargs"]["max_token_freq"], int) or config["vocab_kwargs"]["max_token_freq"] is None
    if config["vocab_kwargs"]["max_token_freq"] is not None:
        assert config["vocab_kwargs"]["max_token_freq"] > config["vocab_kwargs"]["min_token_freq"]
    assert isinstance(config["vocab_kwargs"]["ngrams"][0], int)
    assert isinstance(config["vocab_kwargs"]["ngrams"][1], int)
    assert config["vocab_kwargs"]["ngrams"][0] >= config["vocab_kwargs"]["ngrams"][1]

def load_configuration(args,
                       base_outdir,
                       configuration_type):
    """
    Load the experiment configuration JSON file. Check
    that required fields are present and valid data types.

    Args:
        args (argparse Object): Command-line argument holder.
        base_outdir (str): Base output directory (e.g. data/results/domain_transfer/ or models/)
        configuration_type (str): One of "model","domain_transfer", or "learning_curve"

    Returns:
        config (dict): Dictionary of experiment parameters.
    """
    ## Load the configuration JSON
    with open(args.config_filepath,"r") as the_file:
        config = json.load(the_file)
    ## Validate Configuration
    _ = _validate_config_params(config,
                                configuration_type)
    ## Cache Config, Setup Output Directory
    outdir = _cache_config_and_setup_dirs(config,
                                          base_outdir)
    config["outdir"] = outdir
    return config

def _assign_value_to_bin(value, bins):
    """
    Assign a value to a bin index based on a set of bins

    Args:
        value (numeric): Value for assignment
        bins (list): Bin boundaries. [a,b) bins will be considered
    
    Returns:
        b (int): Index of bin boundaries where value lies.
    """
    if value < bins[0]:
        return -1
    b = 0
    for bstart, bend in zip(bins[:-1], bins[1:]):
        if value >= bstart and value < bend:
            return b
        b += 1
    return b

def _generate_stratified_sample(x, y, n, bins=100):
    """
    Given a population represented by x, select a subset of a population
    represented by y that looks like population x

    Args:
        x (array): Representative values (numeric) for population x
        y (array): Representative values (numeric) for population y
        n (int): How many samples to select from population y
        bins (int): How fine the binning should be for quantifying the population of x
    
    Returns:
        y_sample (list): List of integer indices from sample y that will give
                         a representative sample similar to that parameterized by x
    """
    ## Bin The Input Sample To Match (x)
    x_counts, x_bins = np.histogram(x, bins=bins)
    ## Compute Selection Probability, With 0 probability assigned to below and above extrema
    p_x = np.array([0] + list(x_counts / x_counts.sum()) + [0])
    ## Bin the Y Values, Adding 1 to Account for Extrema Probabilities
    y_binned = np.array(list(map(lambda v: _assign_value_to_bin(v, x_bins), y))) + 1
    y_p_select = p_x[y_binned]
    ## Normalize Sample Probabilities
    y_p_select = y_p_select / y_p_select.sum()
    ## Sample y
    m = len(y)
    y_sample = np.random.choice(m, size=n, replace=False, p=y_p_select)
    ## Sort Sample
    y_sample = sorted(y_sample)
    return y_sample

def _sample_by_age_and_gender(metadata_df,
                              target_disorder,
                              genders=["M","F"],
                              random_state=42):
    """
    Identify a matched sample of users in a target disorder set based 
    on data set split, age, and gender

    Args:
        metadata_df (pandas DataFrame): Label data
        target_disoder (str): Name of the disorder being modeled
        genders (list): List of genders to split on.
        random_state (int): Random seed for sampling
    
    Returns:
        matched_data (pandas DataFrame): Balanced sample of data
    """    
    ## Set Random Sample Seed
    np.random.seed(random_state) 
    ## Train/Dev/Test Split Preservation
    metadata_df = metadata_df.copy()
    if "split" in metadata_df.columns and len(metadata_df["split"].unique()) > 1:
        splits = metadata_df["split"].unique()
    else:
        splits = ["all"]
        metadata_df["split"] = "all"
    ## Separate Groups
    target_disorder_group = metadata_df.loc[metadata_df[target_disorder] == target_disorder]
    control_group = metadata_df.loc[metadata_df[target_disorder] == "control"]
    ## Sample Population
    matched_data = []
    for split in splits:
        ## Cycle Through Genders
        for gender in genders:
            ## Isolate Pool of Users
            gender_target_disorder_pool = target_disorder_group.loc[(target_disorder_group["gender"]==gender)&
                                                                    (target_disorder_group["split"]==split)]
            gender_control_pool = control_group.loc[(control_group["gender"]==gender)&
                                                    (control_group["split"]==split)]
            ## Sample Control Population Based on Age
            gender_control_sample = _generate_stratified_sample(gender_target_disorder_pool["age"].values,
                                                                gender_control_pool["age"].values,
                                                                n = len(gender_target_disorder_pool),
                                                                bins = 100)
            matched_data.append(gender_control_pool.iloc[gender_control_sample])
            matched_data.append(gender_target_disorder_pool)
    matched_data = pd.concat(matched_data).reset_index(drop=True).copy()
    return matched_data

def _select_matched_user_pairs(metadata_df,
                               target_disorder):
    """
    Assuming matched pairs already exist in a metadata data set, this selects the appropriate 
    pairs for a target disorder. Primarily relevant for the Multi-Task Learning data set

    Args:
        metadata_df (pandas DataFrame): Labels for the data set
        target_disorder (str): Mental health disorder being modeled
    
    Returns:
        matched_data (pandas DataFrame): Balanced, and controlled, pre-specified sampled of the data
    """
    ## Separate Groups
    target_disorder_group = metadata_df.loc[metadata_df[target_disorder] == target_disorder]
    control_group = metadata_df.loc[metadata_df.user_id_str.isin(target_disorder_group.user_id_str_matched)]
    ## Concatenate
    matched_data = pd.concat([target_disorder_group, control_group]).reset_index(drop=True).copy()
    return matched_data

def _isolate_appropriate_controls(metadata_df,
                                  dataset,
                                  target_disorder,
                                  random_state=42):
    """
    Isolate appropriate control samples for a target disorder amongst a given
    data set.

    Args:
        metadata_df (pandas DataFrame): Labels for a data set
        dataset (str): Colloquial name of the data set (e.g. rsdd, clpsych_deduped)
        target_disorder (str): Mental health disorder being modeled
        random_state (int): Seed for any sampling being done
    
    Returns:
        matched_data (pandas DataFrame): Balanced, and controlled, sample of the data if applicable.
    """
    ## Make A Copy of the Data Set
    metadata_df = metadata_df.copy()
    ## Drop Rows With Null Target Disorder (e.g. User is not control and not target class)
    metadata_df = metadata_df.dropna(subset=[target_disorder]).reset_index(drop=True)
    ## Identify Matched Groups
    if dataset in ["clpsych","clpsych_deduped"]:
        ## Sample Age + Gender
        matched_data = _sample_by_age_and_gender(metadata_df,
                                                 target_disorder,
                                                 random_state=random_state)
    elif dataset in ["multitask"]:
        ## Get Matched User Pairs
        matched_data = _select_matched_user_pairs(metadata_df,
                                                  target_disorder)
    elif dataset in ["merged"]:
        ## Separate Data Sets Within Merged
        clpsych_metadata_df = metadata_df.loc[metadata_df.datasets.str.contains("clpsych")]
        multitask_metadata_df = metadata_df.loc[metadata_df.datasets.str.contains("multitask")]
        ## Sample CLPsych
        matched_clpsych_data = _sample_by_age_and_gender(clpsych_metadata_df,
                                                         target_disorder,
                                                         random_state=random_state)
        ## Multitask
        matched_multitask_data = _select_matched_user_pairs(multitask_metadata_df,
                                                            target_disorder)
        ## Merge
        matched_data = pd.concat([matched_clpsych_data, matched_multitask_data]).reset_index(drop=True).copy()
    elif dataset == "smhd":
        ## Focus on Train/Dev/Test Splits (e.g. ignore relaxed)
        matched_data = metadata_df.loc[metadata_df["split"]!="relaxed"].reset_index(drop=True).copy()
    else:
        ## No Additional Sampling for RSDD
        matched_data = metadata_df
    return matched_data


def load_dataset_metadata(dataset,
                          target_disorder,
                          random_state=42):
    """
    Load metadata for a particular data set

    Args:
        dataset (str): Canonical name of the dataset
        target_disorder (str): Name of the target disorder
        random_state (int): Seed for any control sampling
    
    Returns:
        metadata_df (pandas DataFrame): Metadata dataframe, sorted
                        by the "user_id_str" in ascending order.
    """
    ## Get Key
    dpath_key = [d for d in DATA_PATHS.keys() if dataset in d][0]
    dpath = DATA_PATHS[dpath_key]
    ## Load Metadata
    metafile = "{}{}_metadata.csv".format(dpath["path"], dataset)
    metadata_df = pd.read_csv(metafile, low_memory=False)
    ## Isolate Appropriate Controls (If Applicable)
    metadata_df = _isolate_appropriate_controls(metadata_df,
                                                dataset,
                                                target_disorder,
                                                random_state=random_state)
    ## Get Absolute Path
    metadata_df["source"] = metadata_df["source"].map(os.path.abspath)
    return metadata_df

def _rebalance(metadata_df,
               target_disorder,
               target_class_ratio,
               random_seed=42):
    """
    Args:
        metadata_df (pandas DataFrame):  Labels for a data set
        target_disorder (str): Name of the target disorder (e.g. "depression", "ptsd")
        target_class_ratio (list): [Target Disorder, Control] Ratio
        random_seed (int): Sample Seed
    
    Returns:
        metadata_df (pandas DataFrame): Rebalanced data set
    """
    ## Copy the Original Data
    metadata_df = metadata_df.copy()
    ## Identify Existing Class Distribution
    class_distribution = metadata_df[target_disorder].value_counts()
    ## Target Sizes
    target_control_size = int(class_distribution[target_disorder] * target_class_ratio[1] / target_class_ratio[0])
    ## Downsample Relative to Target Mental Health Disorder Size
    if class_distribution["control"] >= target_control_size: ## Case 1: Keep Target Class Fixed
        control_sample = metadata_df.loc[metadata_df[target_disorder]=="control"].sample(n=target_control_size,
                                                                                      random_state=random_seed,
                                                                                      replace=False)
        metadata_df = metadata_df.loc[metadata_df[target_disorder]==target_disorder].append(control_sample).reset_index(drop=True)
    else: ## Case 2: Downsample Everything So That Ratio Is Preserved
        n_target = class_distribution[target_disorder]
        while (n_target * target_class_ratio[1]) > class_distribution["control"]:
            n_target -= 1
        n_control = target_class_ratio[1] * n_target
        control_sample = metadata_df.loc[metadata_df[target_disorder]=="control"].sample(n=n_control,
                                                                                      random_state=random_seed,
                                                                                      replace=False)
        target_sample = metadata_df.loc[metadata_df[target_disorder]==target_disorder].sample(n=n_target,
                                                                                              random_state=random_seed,
                                                                                              replace=False)
        metadata_df = control_sample.append(target_sample).reset_index(drop=True)
    return metadata_df

def _downsample(metadata_df,
                downsample_size,
                random_seed=42):
    """
    Downsample a data set

    Args:
        metadata_df (pandas DataFrame): Dataframe of labels, files
        downsample_size (int): Size of the desired downsampled dataset
        random_seed (int): Random seed for sampling

    Returns:
        metadata_df (pandas DataFrame): Downsampled DataFrame of labels, files
    """
    ## Make a Copy
    metadata_df = metadata_df.copy()
    ## Identify Current size
    N = len(metadata_df)
    downsample_size = min(downsample_size, N)
    ## Sample
    metadata_df = metadata_df.sample(n=downsample_size,
                                     random_state=random_seed,
                                     replace=False)
    ## Re-index
    metadata_df = metadata_df.reset_index(drop=True).copy()
    return metadata_df

def split_data(config,
               metadata_df,  
               downsample=False,
               downsample_size=None,
               rebalance=False,
               class_ratio=None,
               downsample_test=False,
               downsample_size_test=None,
               rebalance_test=False,
               class_ratio_test=None):
    """
    Creates a held-out test set and then identifies K-fold
    splits within the training set for hyperparameter optimization.

    Args:
        config (dict): Dataset parameters (e.g. random_seed, stratified, kfolds)
        metadata_df (pandas DataFrame): Metadata DataFrame for data set
        downsample (bool): If True, downsample the data after any class-rebalancing
        downsample_size (int): If downsample is True, this controls the downsample size
        rebalance (bool): If True, rebalance the classes
        class_ratio (list or None): If rebalance is True, this specifies the desired class ratio
                                  of target disorder to control
        ** _test (various): Same meaning as arguments above, but used for sampling test data

    Returns:
        label_dictionaries (dict): Mapping between train, test, k-fold and (user file, label)
    """
    ## Random Seed
    np.random.seed(config["random_seed"])
    ## Make Copy of Metadata
    metadata_df = metadata_df.copy()
    ## Check for Predefined Splits
    if "split" in metadata_df.columns and not np.any(metadata_df["split"].isnull()):
        ## Get Test DF
        test_df = metadata_df.loc[metadata_df["split"]=="test"]
        test_df = test_df.reset_index(drop=True).copy()
        ## Get Train DF
        train_df = metadata_df.loc[metadata_df["split"]=="train"]
        train_df = train_df.reset_index(drop=True).copy()
        ## Get Dev DF if it exists
        if "dev" in metadata_df["split"].values:
            dev_df = metadata_df.loc[metadata_df["split"]=="dev"]
            dev_df = dev_df.reset_index(drop=True).copy()
        else:
            dev_df = train_df.copy()
    else:
        ## Append Weights for Sampling
        if config["stratified"]:
            p = metadata_df[config["target_disorder"]].value_counts(normalize=True)[config["target_disorder"]]
            weights = np.array(list(map(lambda i: p if i==config["target_disorder"] else 1-p,
                                        metadata_df[config["target_disorder"]])))
        else:
            weights = np.ones(metadata_df.shape[0])
        metadata_df["sample_weight"] = weights / weights.sum()
        ## Sample Test Data
        test_df = metadata_df.sample(frac = config["test_size"],
                                     random_state = config["random_seed"],
                                     replace = False,
                                     weights = metadata_df["sample_weight"])
        ## Isolate Remaining Training Data
        train_df = metadata_df.loc[~metadata_df.index.isin(test_df.index)]
        train_df = train_df.reset_index(drop=True).copy()
        ## Set Dev DF to Copy of the Training Data for Splitting
        dev_df = train_df.copy()
    ## Rebalancing (Train and Dev Only)
    if rebalance:
        train_df = _rebalance(train_df, config["target_disorder"], class_ratio, config["random_seed"])
        dev_df = _rebalance(dev_df,  config["target_disorder"], class_ratio, config["random_seed"])
    if rebalance_test:
        test_df = _rebalance(test_df, config["target_disorder"], class_ratio_test, config["random_seed"])
    if downsample:
        train_df = _downsample(train_df, downsample_size, config["random_seed"])
        dev_df = _downsample(dev_df, downsample_size, config["random_seed"])
    if downsample_test:
        test_df = _downsample(test_df, downsample_size_test, config["random_seed"])
    ## K-Fold Split
    if config["stratified"]:
        splitter = StratifiedKFold(n_splits=config["kfolds"],
                                   shuffle=True,
                                   random_state=config["random_seed"])
    else:
        splitter = KFold(n_splits=config["kfolds"],
                         shuffle=True,
                         random_state=config["random_seed"])
    ## Initialize Splitters
    train_splitter = splitter.split(train_df.index.tolist(),
                                    train_df[config["target_disorder"]].values)
    dev_splitter = splitter.split(dev_df.index.tolist(),
                                  dev_df[config["target_disorder"]].values)
    ## Create Split Dictionaries
    train_splits = {}
    for k, ((train_, _),(_, dev_)) in enumerate(zip(train_splitter, dev_splitter)):
        train_splits[k+1] = {
            "train": train_df.loc[train_].set_index("source")[config["target_disorder"]].to_dict(),
            "dev": dev_df.loc[dev_].set_index("source")[config["target_disorder"]].to_dict(),
            "test": None,
                }
    ## Create Standardized Label Dictionaries
    label_dictionaries = {
        "train":train_splits,
        "test":{1:  {
                     "train": None,
                     "dev": None,
                     "test": test_df.set_index("source")[config["target_disorder"]].to_dict()
                    }
                }
    }
    return label_dictionaries


def fit_model(config,
              data_key,
              train_labels,
              dev_labels,
              output_dir,
              grid_search,
              grid_search_kwargs,
              cache_model=True,
              cache_predictions=True,
              train_date_boundaries={},
              dev_date_boundaries={},
              train_samples={},
              dev_samples={},
              drop_null_train=False,
              drop_null_dev=False):
    """
    Fit a single mental health classifier

    Args:
        config (dict): Model configuration parameters (bare minimum)
        data_key (Any): Key for the prediction dictionary (e.g. the fold number) 
        train_labels (dict): Mapping between filenames and labels for training
        dev_labels (dict): Mapping between filenames and labels for development data
        output_dir (str): Where the trained model and results should be stored
        grid_search (bool): Whether or not to execute a grid search
        grid_search_kwargs (dict): Arguments to pass to grid search if running
        cache_model (bool): If True, cache the model
        cache_predictions (bool): If True, cache the predictions to disk
        train_date_boundaries (dict): {"min_date":val, "max_date"} Date boundaries
        dev_date_boundaries (dict): {"min_date":val, "max_date"} Date boundaries
        train_samples (dict): Keys "n_samples" and "randomized"
        dev_samples (dict): Keys "n_samples" and "randomized"
        drop_null_train (bool): Whether to ignore samples without features during training
        drop_null_dev (bool): Whether to ignore samples without features during prediction

    Returns:
        predictions (dict): {data_key : {"train":train_pred, "dev":dev_pred}} where
                             train_pred and dev_pred are dictionary mappings between
                             filenames and user label prediction
        model (MentalHealthClassifer): The fit model
    """
    ## Initialize Base Model
    model = MentalHealthClassifier(target_disorder=config["target_disorder"],
                                   model=config.get("model"),
                                   feature_selector=config.get("feature_selector"),
                                   vocab_kwargs=config.get("vocab_kwargs"),
                                   preprocessing_kwargs=config.get("preprocessing_kwargs"),
                                   model_kwargs=config.get("model_kwargs"),
                                   feature_selection_kwargs=config.get("feature_selection_kwargs"),
                                   jobs=config.get("jobs"),
                                   random_state=config.get("random_seed"),
                                   min_date=train_date_boundaries.get("min_date"),
                                   max_date=train_date_boundaries.get("max_date"),
                                   n_samples=train_samples.get("n_samples"),
                                   randomized=train_samples.get("randomized"),
                                   drop_null=drop_null_train)
    ## Enumerate Training/Test Files
    train_files = list(train_labels.keys())
    if dev_labels is not None:
        dev_files = list(dev_labels.keys())
    else:
        dev_files = None
    ## Option 1: Fit With Grid Search
    if grid_search:
        ## Development Files to Use In Grid Search Development
        if grid_search_kwargs["dev_frac"] > 0 and dev_labels is not None:
            np.random.seed(config.get("random_seed"))
            gs_dev_files = np.random.choice(dev_files,
                                            size=int(len(pred_files) * grid_search_kwargs["dev_frac"]),
                                            replace=False)
            gs_dev_dict = dev_labels
        else:
            gs_dev_files = None
            gs_dev_dict = None
        ## Run Grid Search and Fit
        model, train_pred = model.fit_with_grid_search(train_files=train_files,
                                                       train_label_dict=train_labels,
                                                       dev_files=gs_dev_files,
                                                       dev_label_dict=gs_dev_dict,
                                                       config=grid_search_kwargs["config"],
                                                       test_size=grid_search_kwargs["test_size"],
                                                       dev_in_vocab=grid_search_kwargs["dev_in_vocab"],
                                                       score_func=SCORE_FUNCTIONS[grid_search_kwargs["score_func"]],
                                                       use_multiprocessing=grid_search_kwargs["use_multiprocessing"],
                                                       cache_results=True,
                                                       return_training_preds=True)
    else:
        ## Option 2: Fit With Default Hyperparameters
        model, train_pred = model.fit(train_files=train_files,
                                      label_dict=train_labels,
                                      return_training_preds=True)
    ## Make Test Predictions
    if dev_files is not None:
        dev_pred = model.predict(test_files=dev_files,
                                 min_date=dev_date_boundaries.get("min_date"),
                                 max_date=dev_date_boundaries.get("max_date"),
                                 n_samples=dev_samples.get("n_samples"),
                                 randomized=dev_samples.get("randomized"),
                                 drop_null=drop_null_dev)
    else:
        dev_pred = {}
    ## Compile Results
    predictions = {data_key : {"train":train_pred, "dev":dev_pred}}
    ## Cache Model (if Desired)
    if cache_model:
        _ = model.dump("{}model.joblib".format(output_dir))
    ## Cache Predictions
    if cache_predictions:
        with gzip.open("{}predictions.tar.gz".format(output_dir), "wt", encoding="utf-8") as the_file:
            json.dump(predictions, the_file)
    return predictions, model