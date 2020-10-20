
##########################
### Configuration
##########################

## Grid Username
USERNAME = "kharrigian"

## Run Prefix (Unique to make processing results after easier)
EXPERIMENT_PREFIX = "TemporalEffectsBalanced"

## Choose Experiment Template
BASE_EXPERIMENT_CONFIG = "camera_ready/temporal_effects.json"

## Dataset-specific Parameters
DATASET_PARAMETERS = {
        "clpsych_deduped":{
                    "test_sets":["clpsych_deduped","multitask","rsdd","smhd","wolohan"],
                    "date_boundaries":["2012-01-01","2013-01-01","2013-12-01"],
                    "downsample":False,
                    "downsample_size":50,
                    "rebalance":True,
                    "class_ratio":[1, 1],
                    "use_multiprocessing_grid_search":False,
                    "post_sampling":{"n_samples":200,"randomized":True},
                    "min_posts":200,
                  },
        "multitask":{
                    "test_sets":["clpsych_deduped","multitask","rsdd","smhd","wolohan"],
                    "date_boundaries":["2013-01-01","2014-01-01","2015-01-01","2016-01-01"],
                    "downsample":False,
                    "downsample_size":50,
                    "rebalance":True,
                    "class_ratio":[1,1],
                    "use_multiprocessing_grid_search":False,
                    "post_sampling":{"n_samples":200,"randomized":True},
                    "min_posts":200,
                   },
        "wolohan":{
                    "test_sets":["clpsych_deduped","multitask","rsdd","smhd","wolohan"],
                    "date_boundaries":["2015-01-01","2016-01-01","2017-01-01","2018-01-01","2019-01-01","2020-01-01"],
                    "downsample":False,
                    "downsample_size":50,
                    "rebalance":True,
                    "class_ratio":[1,1],
                    "use_multiprocessing_grid_search":False,
                    "post_sampling":{"n_samples":100,"randomized":True},
                    "min_posts":100
        },
        "rsdd":{
                "test_sets":["clpsych_deduped","multitask","rsdd","wolohan"],
                "date_boundaries":["2012-01-01","2013-01-01","2014-01-01","2015-01-01","2016-01-01","2017-01-01"],
                "downsample":False,
                "downsample_size":50,
                "rebalance":True,
                "class_ratio":[1,1],
                "use_multiprocessing_grid_search":False,
                "post_sampling":{"n_samples":100,"randomized":True},
                "min_posts":100,
        },
        "smhd":{
                "test_sets":["clpsych_deduped","multitask","smhd","wolohan"],
                "date_boundaries":["2012-01-01","2013-01-01","2014-01-01","2015-01-01","2016-01-01","2017-01-01","2018-01-01"],
                "downsample":False,
                "downsample_size":50,
                "rebalance":True,
                "class_ratio":[1,1],
                "use_multiprocessing_grid_search":False,
                "post_sampling":{"n_samples":100,"randomized":True},
                "min_posts":100,
        }
}

## Meta Parameters
ALLOW_MIXED_TIME_PERIODS = False

##########################
### Imports
##########################

## Standard Library
import os
import sys
import json
import subprocess
from time import sleep
from uuid import uuid4
from copy import deepcopy
from datetime import datetime
from functools import partial
from collections import Counter

## External
import joblib
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack

## Local
from mhlib.model import train
from mhlib.util.logging import initialize_logger
from mhlib.util.multiprocessing import MyPool as Pool
from mhlib.model.data_loaders import LoadProcessedData

##########################
### Globals
##########################

## Initialize Logger
LOGGER = initialize_logger()

## Root Repository Directory
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)) + "/../../../"

## Critical Experiment Directories
CONFIG_PATH = f"{ROOT_PATH}configurations/experiments/"
HYPERPARAMETER_PATH = f"{ROOT_PATH}configurations/hyperparameter_search/" 
TEMP_DIR = f"{ROOT_PATH}temp_{str(uuid4())}/"
MODELS_DIR = f"{TEMP_DIR}models/"

##########################
### Functions
##########################
 
def load_base_config(config):
    """
    Load the configuration template that will be used for all
    experiments.

    Args:
        config (str): Name of the configuration file (ignoring directory)
    
    Returns:
        config_data (dict): Dictionary of configuration parameters
    """
    config_file = f"{CONFIG_PATH}{config}"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Could not find config file: '{config_file}'")
    with open(config_file, "r") as the_file:
        config_data = json.load(the_file)
    return config_data

def load_hyperparameter_config(config):
    """
    Load the hyperparameter-search template that will be used for all
    experiments (if turned on in the regular config)

    Args:
        config (dict): Dictionary of configuration parameters
    
    Returns:
        hyperparameters (list of dict): Grid search parameters
    """
    hyperparameter_file = "{}{}".format(HYPERPARAMETER_PATH,
                                        os.path.basename(config["grid_search_kwargs"]["config"]))
    if not os.path.exists(hyperparameter_file):
        raise FileNotFoundError(f"Could not find grid search file: '{hyperparameter_file}'")
    with open(hyperparameter_file, "r") as the_file:
        hyperparameters = json.load(the_file)
    return hyperparameters

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
                                date_boundaries):
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
    date_range_bins = pd.to_datetime(date_boundaries)
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

def cache_date_distributions(config,
                             dataset_parameters):
    """

    """
    ## Store Filepaths
    date_distributions = {}
    ## Get Distributions for Each Data Set
    for dataset, dataset_params in dataset_parameters.items():
        LOGGER.info(f"Getting Date Distribution for {dataset}")
        ## Load Metadata
        metadata = train.load_dataset_metadata(dataset,
                                               config["target_disorder"],
                                               config["random_seed"])
        ## Get Temporal Breakdown
        filenames, X_time = retrieve_temporal_breakdown(metadata,
                                                        config,
                                                        dataset_params["date_boundaries"])
        ## Cache
        cache_file = f"{TEMP_DIR}{dataset}_date_distribution.joblib"
        _ = joblib.dump({"X_time":X_time,
                         "filenames":filenames,
                         "date_boundaries":dataset_params["date_boundaries"],
                         "target_disorder":config["target_disorder"],
                         "random_seed":config["random_seed"]},
                        cache_file)
        date_distributions[dataset] = cache_file
    return date_distributions

def setup_experiments(config,
                      dataset_parameters,
                      date_distributions):
    """
    Create list of experiments, manipulating parameters in the base
    configuration to be dataset specific (e.g. sample size, balance)

    Args:
        config (dict): Dictionary of configuration parameters
        dataset_parameters (dict): Configuration parameters to update
                                   for each data set (head of this file)
        date_distributions (dict): Mapping between each dataset and cached
                                   date distribution file
    
    Returns:
        experiments (list): List of configuration dictionaries for each 
                            domain transfer experiment
    """
    ## Cache of Experiments
    experiments = []
    for train_set, train_set_parameters in dataset_parameters.items():
        for test_set in train_set_parameters["test_sets"]:
            combo_config = deepcopy(config)
            combo_config["train_data"] = train_set
            combo_config["test_data"] = test_set
            if train_set == test_set and ALLOW_MIXED_TIME_PERIODS:
                combo_config["mixed_time_windows"] = True
            else:
                combo_config["mixed_time_windows"] = False
            combo_config["date_distributions"] = {
                                                    "train":date_distributions[train_set],
                                                    "test":date_distributions[test_set]
                                                 }
            combo_config["post_sampling"] = {
                                    "train":dataset_parameters[train_set]["post_sampling"].copy(),
                                    "test":dataset_parameters[test_set]["post_sampling"].copy()
            }
            combo_config["min_posts"] = {
                "train":dataset_parameters[train_set]["min_posts"],
                "test":dataset_parameters[test_set]["min_posts"]
            }
            combo_config["downsample"] = {"train":dataset_parameters[train_set]["downsample"],
                                          "test":dataset_parameters[test_set]["downsample"]}
            combo_config["date_boundaries"] = {"train":dataset_parameters[train_set]["date_boundaries"],
                                               "test":dataset_parameters[test_set]["date_boundaries"]}
            combo_config["downsample_size"] = {"train":dataset_parameters[train_set]["downsample_size"],
                                               "test":dataset_parameters[test_set]["downsample_size"]}
            combo_config["rebalance"] = {"train":dataset_parameters[train_set]["rebalance"],
                                          "test":dataset_parameters[test_set]["rebalance"]}
            combo_config["class_ratio"] = {"train":dataset_parameters[train_set]["class_ratio"],
                                           "test":dataset_parameters[test_set]["class_ratio"]}            
            combo_config["grid_search_kwargs"]["use_multiprocessing"] = dataset_parameters[train_set]["use_multiprocessing_grid_search"]

            combo_config["experiment_name"] = f"{EXPERIMENT_PREFIX}_TRAIN_{train_set}-TEST_{test_set}"
            experiments.append(combo_config)
    return experiments

def get_bash_script(experiment_name,
                    exp_config_file,
                    train_date_distribution_file,
                    test_date_distribution_file
                    ):
    """
    Generate code for a bash script that will be called
    by qsub to schedule a domain transfer experiment on
    the CLSP grid

    Args:
        experiment_name (str): Name of the experiment
        exp_config_file (str): Path to the experiment configuration file
    
    Returns:
        script (str): Code that will be written to a bash file
    """
    script = """
    #!/bin/bash
    #$ -cwd
    #$ -S /bin/bash
    #$ -m eas
    #$ -e /home/kharrigian/gridlogs/python/{}.err
    #$ -o /home/kharrigian/gridlogs/python/{}.out
    #$ -pe smp 8
    #$ -l 'gpu=0,mem_free=12g,ram_free=12g'

    ## Move to Home Directory (Place Where Virtual Environments Live)
    cd /home/kharrigian/
    ## Activate Conda Environment
    source .bashrc
    conda activate mental-health
    ## Move To Run Directory
    cd /export/fs03/a08/kharrigian/mental-health/
    ## Run Script
    python ./scripts/experiment/temporal_effects/temporal_effects.py {} --train_date_distribution {} --test_date_distribution {} --models_dir {}
    """.format(experiment_name,
               experiment_name,
               exp_config_file,
               train_date_distribution_file,
               test_date_distribution_file,
               MODELS_DIR
               )
    return script

def start_job(experiment_config,
              hyperparameter_config):
    """
    Schedule a domain transfer job

    Args:
        experiment_config (dict): Individual domain transfer experiment configuration
        hyperparameter_config (dict): Dictionary of grid-search parameters
    
    Returns:
        job_id (int): Id of the job scheduled by qsub on the CLSP grid
        exp_name (str): Name of the experiment associated with the job
        start_time (datetime): Start time of the experiment
        exp_config_file (str): Path to the experiment configuration file to be deleted afterwards
        exp_hyperparameter_file (str): Path to the experiment hyperparameter search file to be
                                       deleted afterwards
    """
    ## Start Experiment
    exp_name = experiment_config.get("experiment_name")
    start_time = datetime.now()
    clean_start = datetime.strftime(start_time, "%m-%d-%Y %I:%M:%S%p")
    LOGGER.info("~"*100 + f"\nStarting Experiment: {exp_name} at {clean_start}\n" + "~"*100)
    ## Write Experiment Config Files
    exp_config_file = "{}{}.json".format(CONFIG_PATH, exp_name)
    exp_hyperparameter_file = "{}{}.json".format(HYPERPARAMETER_PATH, exp_name)
    experiment_config["grid_search_kwargs"]["config"] = exp_hyperparameter_file
    with open(exp_config_file, "w") as the_file:
        json.dump(experiment_config, the_file)
    with open(exp_hyperparameter_file, "w") as the_file:
        json.dump(hyperparameter_config, the_file)
    ## Write Bash Script
    bash_script = get_bash_script(exp_name,
                                  exp_config_file,
                                  experiment_config["date_distributions"]["train"],
                                  experiment_config["date_distributions"]["test"])
    job_file = f"{TEMP_DIR}{exp_name}.sh"
    with open(job_file, "w") as the_file:
        the_file.write("\n".join([i.lstrip() for i in bash_script.split("\n")]))
    ## Schedule Job
    qsub_call = f"qsub {job_file}"
    job_id = subprocess.check_output(qsub_call, shell=True)
    job_id = int(job_id.split()[2])
    return job_id, exp_name, start_time, exp_config_file, exp_hyperparameter_file

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
                           wait_time=30):
    """
    Sleep until all jobs scheduled have been completed

    Args:
        scheduled_jobs (list): Output of schedule_jobs
    
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
            LOGGER.info("Jobs Complete: {}".format(newly_completed_jobs))
            n_jobs_complete = len(complete_jobs)
            n_jobs_remaining = len(scheduled_jobs) - n_jobs_complete
            sleep_count = 0

def schedule_jobs(experiments,
                  base_hyperparameter_search):
    """
    Go through all experiments and schedule them for execution
    on the CLSP grid

    Args:
        experiments (list of dict): List of experiment configurations
        base_hyperparameter_search (dict): Grid search parameters to be used (if specified
                                          by the experiment configuration)
        
    Returns:
        scheduled_jobs (list of tuples): Scheduled jobs and metadata
    """
    executed_jobs = []
    for n, next_experiment in enumerate(experiments):
        LOGGER.info("Scheduling Experiment {}/{}".format(n+1, len(experiments)))
        job_id, job_name, job_start_time, job_config_file, job_hyperparameter_config_file = start_job(next_experiment,
                                                                                                      base_hyperparameter_search)
        executed_jobs.append([job_id, job_name, job_start_time, job_config_file, job_hyperparameter_config_file])
        _ = hold_for_complete_jobs([[job_id, job_name, job_start_time, job_config_file, job_hyperparameter_config_file]])
    return executed_jobs

def run_teardown(executed_jobs):
    """
    Remove temporary bash and configuration files used
    as part of the job array

    Args:
        scheduled_jobs (list): Output from schedule_jobs
    
    Returns:
        None
    """
    for s in executed_jobs:
        _ = os.system("rm {}".format(s[3]))
        _ = os.system("rm {}".format(s[4]))
        _ = os.system("rm {}{}.sh".format(TEMP_DIR, s[1]))
    _ = os.system(f"rm -rf {TEMP_DIR}")

def main():
    """
    Main Program. Run all domain transfer experiments
    requested in the configuration of this script

    Args:
        None
    
    Returns:
        None
    """
    ## Create Temp Directory
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    ## Load Configuration
    base_config = load_base_config(BASE_EXPERIMENT_CONFIG)
    ## Load Hyperparameter Search Configuration
    base_hyperparameter_search = load_hyperparameter_config(base_config)
    ## Setup Cached Date Distributions
    date_distributions = cache_date_distributions(base_config,
                                                  DATASET_PARAMETERS)
    ## Setup Experiments
    experiments = setup_experiments(base_config,
                                    DATASET_PARAMETERS,
                                    date_distributions)
    LOGGER.info("Identified {} experiments.".format(len(experiments)))
    ## Schedule Experiments
    executed_jobs = schedule_jobs(experiments,
                                  base_hyperparameter_search)
    ## Run Teardown
    _ = run_teardown(executed_jobs)
    ## Procedure Complete
    LOGGER.info("All Experiments Complete!")

##########################
### Execution
##########################
 
if __name__ == "__main__":
    _ = main()