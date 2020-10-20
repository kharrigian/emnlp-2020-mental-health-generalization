
##########################
### Configuration
##########################

## Grid Username
USERNAME = "kharrigian"

## Run Prefix (Unique to make processing results after easier)
EXPERIMENT_PREFIX = "BalancedOnly-CameraReady"

## Choose Experiment Template
BASE_EXPERIMENT_CONFIG = "camera_ready/domain_transfer.json"

## Dataset-specific Parameters
DATASET_PARAMETERS = {
        "clpsych_deduped":{
                    "test_sets":["clpsych_deduped"],
                    "downsample":True,
                    "downsample_size":580,
                    "rebalance":True,
                    "class_ratio":[1, 1],
                    "use_multiprocessing_grid_search":False,
                    "date_boundaries":{"min_date":"2011-01-01","max_date":"2013-12-01"},
                    "post_sampling":{"n_samples":None,"randomized":False},
                  },
        "multitask":{
                    "test_sets":["multitask"],
                    "downsample":False,
                    "downsample_size":580,
                    "rebalance":True,
                    "class_ratio":[1,1],
                    "use_multiprocessing_grid_search":False,
                    "date_boundaries":{"min_date":"2013-01-01","max_date":"2016-01-01"},
                    "post_sampling":{"n_samples":None,"randomized":False},
                   },
        "wolohan":{
                    "test_sets":["wolohan"],
                    "downsample":False,
                    "downsample_size":580,
                    "rebalance":True,
                    "class_ratio":[1,1],
                    "use_multiprocessing_grid_search":False,
                    "date_boundaries":{"min_date":"2014-01-01","max_date":"2020-01-01"},
                    "post_sampling":{"n_samples":None,"randomized":False}
        },
        "rsdd":{
                "test_sets":["rsdd"],
                "downsample":False,
                "downsample_size":580,
                "rebalance":True,
                "class_ratio":[1,1],
                "use_multiprocessing_grid_search":False,
                "date_boundaries":{"min_date":"2008-01-01","max_date":"2017-01-01"},
                "post_sampling":{"n_samples":None,"randomized":False}
        },
        "smhd":{
                "test_sets":["smhd"],
                "downsample":False,
                "downsample_size":580,
                "rebalance":True,
                "class_ratio":[1,1],
                "use_multiprocessing_grid_search":False,
                "date_boundaries":{"min_date":"2010-01-01","max_date":"2018-01-01"},
                "post_sampling":{"n_samples":None,"randomized":False}
        }
}

## Caching
CACHE_MODELS = True
CACHE_PREDICTIONS = True

##########################
### Imports
##########################
 
## Standard Library
import os
import json
import subprocess
from time import sleep
from copy import deepcopy
from datetime import datetime

## Local
from mhlib.util.logging import initialize_logger

## Initialize Logger
logger = initialize_logger()

##########################
### Globals
##########################

## Root Repository Directory
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)) + "/../../../"

## Critical Experiment Directories
CONFIG_PATH = f"{ROOT_PATH}configurations/experiments/"
HYPERPARAMETER_PATH = f"{ROOT_PATH}configurations/hyperparameter_search/" 
TEMP_DIR = f"{ROOT_PATH}temp/"

##########################
### Functions
##########################
 
def load_base_config(config):
    """
    Load the configuration template that will be used for all domain transfer
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
    Load the hyperparameter-search template that will be used for all doamin transfer
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

def setup_experiments(config,
                      dataset_parameters):
    """
    Create list of experiments, manipulating parameters in the base
    configuration to be dataset specific (e.g. sample size, balance)

    Args:
        config (dict): Dictionary of configuration parameters
        dataset_parameters (dict): Configuration parameters to update
                                   for each data set (head of this file)
    
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
            combo_config["post_sampling"] = {"train":dataset_parameters[train_set]["post_sampling"],
                                             "test":dataset_parameters[test_set]["post_sampling"]}
            combo_config["date_boundaries"] = {"train":dataset_parameters[train_set]["date_boundaries"],
                                               "test":dataset_parameters[test_set]["date_boundaries"]}
            combo_config["downsample"] = {"train":dataset_parameters[train_set]["downsample"],
                                          "test":dataset_parameters[test_set]["downsample"]}
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
                    cache_models=CACHE_MODELS,
                    cache_predictions=CACHE_PREDICTIONS):
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
    #$ -l 'gpu=0,mem_free=72g,ram_free=72g'

    ## Move to Home Directory (Place Where Virtual Environments Live)
    cd /home/kharrigian/
    ## Activate Conda Environment
    source .bashrc
    conda activate mental-health
    ## Move To Run Directory
    cd /export/fs03/a08/kharrigian/mental-health/
    ## Run Script
    python ./scripts/experiment/domain_transfer/domain_transfer.py {} {} {}
    """.format(experiment_name,
               experiment_name,
               exp_config_file,
               {True:"--cache_models", False:""}[cache_models],
               {True:"--cache_predictions", False:""}[cache_predictions]
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
    logger.info("~"*100 + f"\nStarting Experiment: {exp_name} at {clean_start}\n" + "~"*100)
    ## Write Experiment Config Files
    exp_config_file = "{}{}.json".format(CONFIG_PATH, exp_name)
    exp_hyperparameter_file = "{}{}.json".format(HYPERPARAMETER_PATH, exp_name)
    experiment_config["grid_search_kwargs"]["config"] = exp_hyperparameter_file
    with open(exp_config_file, "w") as the_file:
        json.dump(experiment_config, the_file)
    with open(exp_hyperparameter_file, "w") as the_file:
        json.dump(hyperparameter_config, the_file)
    ## Write Bash Script
    bash_script = get_bash_script(exp_name, exp_config_file)
    job_file = f"{TEMP_DIR}{exp_name}.sh"
    with open(job_file, "w") as the_file:
        the_file.write("\n".join([i.lstrip() for i in bash_script.split("\n")]))
    ## Schedule Job
    qsub_call = f"qsub {job_file}"
    job_id = subprocess.check_output(qsub_call, shell=True)
    job_id = int(job_id.split()[2])
    return job_id, exp_name, start_time, exp_config_file, exp_hyperparameter_file

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
    scheduled_jobs = []
    for n, next_experiment in enumerate(experiments):
        job_id, job_name, job_start_time, job_config_file, job_hyperparameter_config_file = start_job(next_experiment,
                                                                                                      base_hyperparameter_search)
        scheduled_jobs.append([job_id, job_name, job_start_time, job_config_file, job_hyperparameter_config_file])
    return scheduled_jobs

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

def hold_for_complete_jobs(scheduled_jobs):
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
                logger.info("Domain Transfer jobs still running. Continuing to sleep.")
            sleep(120)
            sleep_count += 1
        else:
            complete_jobs.extend(newly_completed_jobs)
            logger.info("Newly finished jobs: {}".format(newly_completed_jobs))
            n_jobs_complete = len(complete_jobs)
            n_jobs_remaining = len(scheduled_jobs) - n_jobs_complete
            logger.info(f"{n_jobs_complete} jobs complete. {n_jobs_remaining} jobs remaining.")
            sleep_count = 0

def run_teardown(scheduled_jobs):
    """
    Remove temporary bash and configuration files used
    as part of the job array

    Args:
        scheduled_jobs (list): Output from schedule_jobs
    
    Returns:
        None
    """
    for s in scheduled_jobs:
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
    ## Load Configuration
    base_config = load_base_config(BASE_EXPERIMENT_CONFIG)
    ## Load Hyperparameter Search Configuration
    base_hyperparameter_search = load_hyperparameter_config(base_config)
    ## Setup Experiments
    experiments = setup_experiments(base_config, DATASET_PARAMETERS)
    logger.info("Identified {} experiments.".format(len(experiments)))
    ## Schedule Experiments
    scheduled_jobs = schedule_jobs(experiments, base_hyperparameter_search)
    ## Hold Until All Jobs Are Complete
    _ = hold_for_complete_jobs(scheduled_jobs)
    ## Run Teardown
    _ = run_teardown(scheduled_jobs)
    ## Procedure Complete
    logger.info("All Experiments Complete!")

##########################
### Execution
##########################
 
if __name__ == "__main__":
    _ = main()