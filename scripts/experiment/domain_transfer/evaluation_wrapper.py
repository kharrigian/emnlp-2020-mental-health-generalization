
"""
Apply Cross-Validation Models from Domain Transfer to Held-out Test Data
"""

########################
### Configuration
########################

## CLSP Grid Username
USERNAME = "kharrigian"

## Whether to Hold Between Models
HOLD_BETWEEN = False

## Whether to Skip Actual Testing Procedure and Just Do Analysis on Existing Data
SKIP_SCHEDULE = True

## Name for the Run
# EXPERIMENT_LABEL = "BalancedDownsampled"
EXPERIMENT_LABEL = "BalancedOnly"

## Specify Model Directory (Must have Cross-Validation)
# MODEL_DIRS = {
#               "clpsych_deduped":"20201006125542-BalancedDownsampled-CameraReady_TRAIN_clpsych_deduped-TEST_clpsych_deduped",
#               "multitask":"20201006125630-BalancedDownsampled-CameraReady_TRAIN_multitask-TEST_multitask",
#               "rsdd":"20201006140945-BalancedDownsampled-CameraReady_TRAIN_rsdd-TEST_rsdd",
#               "smhd":"20201006143044-BalancedDownsampled-CameraReady_TRAIN_smhd-TEST_smhd",
#               "wolohan":"20201006125836-BalancedDownsampled-CameraReady_TRAIN_wolohan-TEST_wolohan"
# }
MODEL_DIRS = {
              "clpsych_deduped":"20201006143437-BalancedOnly-CameraReady_TRAIN_clpsych_deduped-TEST_clpsych_deduped",
              "multitask":"20201006143506-BalancedOnly-CameraReady_TRAIN_multitask-TEST_multitask",
              "rsdd":"20201006145833-BalancedOnly-CameraReady_TRAIN_rsdd-TEST_rsdd",
              "smhd":"20201006151937-BalancedOnly-CameraReady_TRAIN_smhd-TEST_smhd",
              "wolohan":"20201006144039-BalancedOnly-CameraReady_TRAIN_wolohan-TEST_wolohan"
}

## Specific Condition
CONDITION = "depression"

## Where to Find Split Files for Evaluation
SPLITS = {
        "clpsych_deduped":"./models/20201008222627-CameraReady_CLPsych/splits.json",
        "multitask":"./models/20201008222635-CameraReady_Multitask/splits.json",
        "rsdd":"./models/20201008222650-CameraReady_RSDD/splits.json",
        "smhd":"./models/20201008222646-CameraReady_SMHD/splits.json",
        "wolohan":"./models/20201008222640-CameraReady_Wolohan/splits.json"
}

## Date Ranges
DATE_RANGES = {
        "clpsych_deduped":{
                            "min_date":"2011-01-01",
                            "max_date":"2013-12-01"
        },
        "multitask":{
                    "min_date":"2013-01-01",
                    "max_date":"2016-01-01"
        },
        "rsdd":{
                "min_date":"2008-01-01",
                "max_date":"2017-01-01"
        },
        "smhd":{
                "min_date":"2010-01-01",
                "max_date":"2018-01-01"
        },
        "wolohan":{
                "min_date":"2014-01-01",
                "max_date":"2020-01-01"
        }
}

## Scoring Arguments
PROBABILITY = True
THRESHOLD = 0.5

########################
### Imports
########################

## Standard Library
import os
import sys
import subprocess
from time import sleep
from copy import deepcopy
from datetime import datetime
from glob import glob
from uuid import uuid4

## External Libraries
import numpy as np
import pandas as pd
from sklearn import metrics

## Local
from mhlib.util.logging import initialize_logger

########################
### Globals
########################

## Initialize Logger
LOGGER = initialize_logger()

## Where Results Live
RESULTS_DIR = "./data/results/domain_transfer/"

## Where the Outputs will Live
OUTPUT_DIR = f"./data/results/cache/domain_transfer/test/{EXPERIMENT_LABEL}/"
if not os.path.exists(OUTPUT_DIR):
    _ = os.makedirs(OUTPUT_DIR)

########################
### Helpers
########################

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

def schedule_model_predictions(MODEL_NAME,
                               MODEL_DIR):
    """

    """
    ## Identify Trained Models for Each Fold
    MODELS = sorted(glob(f"{RESULTS_DIR}{MODEL_DIR}/cross_validation/*/*.joblib"))
    MODELS = dict((os.path.dirname(model).split("/")[-1], model) for model in MODELS)
    if SKIP_SCHEDULE:
        return MODELS, []
    ## Output Directory
    MODEL_OUTPUT_DIR = f"{OUTPUT_DIR}{MODEL_NAME}/"
    ## Initialize Script Formatter
    SCRIPT = """
    #!/bin/bash
    #$ -cwd
    #$ -S /bin/bash
    #$ -m eas
    #$ -e /home/kharrigian/gridlogs/python/{}.err
    #$ -o /home/kharrigian/gridlogs/python/{}.out
    #$ -pe smp 8
    #$ -l 'gpu=0,mem_free=32g,ram_free=32g'

    ## Move to Home Directory (Place Where Virtual Environments Live)
    cd /home/kharrigian/
    ## Activate Conda Environment
    source .bashrc
    conda activate mental-health
    ## Move To Run Directory
    cd /export/fs03/a08/kharrigian/mental-health/
    ## Run Script
    python ./scripts/model/evaluate.py \
    --model {} \
    --splits {} \
    --split_keys test 1 test \
    --output_folder {} \
    --min_date {} \
    --max_date {} \
    --prob_threshold {}
    """
    ## Generate Scripts
    SCRIPTS = []
    for dataset, dataset_split in SPLITS.items():
        for fold, model in MODELS.items():
            _script = SCRIPT.format(
                            f"{MODEL_NAME}_{dataset}_{fold}",
                            f"{MODEL_NAME}_{dataset}_{fold}",
                            os.path.abspath(model),
                            os.path.abspath(dataset_split),
                            f"{MODEL_OUTPUT_DIR}{dataset}/{fold}/",
                            DATE_RANGES.get(dataset).get("min_date"),
                            DATE_RANGES.get(dataset).get("max_date"),
                            THRESHOLD
            )
            SCRIPTS.append((dataset, fold, _script))
    ## Create Temporary Script Directory
    TEMP_DIR = "temp_{}/".format(str(uuid4()))
    _ = os.makedirs(TEMP_DIR)
    ## Schedule Jobs
    JOB_IDS = []
    for dataset, fold, _script in SCRIPTS:
        job_file = f"{TEMP_DIR}{MODEL_NAME}_{dataset}_{fold}.sh"
        with open(job_file, "w") as the_file:
            the_file.write("\n".join([i.lstrip() for i in _script.split("\n")]))
        ## Schedule Job
        qsub_call = f"qsub {job_file}"
        job_id = subprocess.check_output(qsub_call, shell=True)
        job_id = int(job_id.split()[2])
        JOB_IDS.append(job_id)
    ## Wait if Desired
    if HOLD_BETWEEN:
        ## Hold For Complete Jobs
        _ = hold_for_complete_jobs(JOB_IDS)
        ## Tear Down Temporary Directory
        _ = os.system("rm -rf {}".format(TEMP_DIR))
    return MODELS, JOB_IDS

def hold_for_complete_jobs(scheduled_jobs,
                           sleep_time=30):
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
    while sorted(complete_jobs) != sorted(scheduled_jobs):
        ## Get Running Jobs
        running_jobs = get_running_jobs()
        ## Look for Newly Completed Jobs
        newly_completed_jobs = []
        for s in scheduled_jobs:
            if s not in running_jobs and s not in complete_jobs:
                newly_completed_jobs.append(s)
        ## Sleep If No Updates, Otherwise Update Completed Job List and Reset Counter
        if len(newly_completed_jobs) == 0:
            if sleep_count % 5 == 0:
                LOGGER.info("Jobs still running. Continuing to sleep.")
            sleep(sleep_time)
            sleep_count += 1
        else:
            complete_jobs.extend(newly_completed_jobs)
            LOGGER.info("Newly finished jobs: {}".format(newly_completed_jobs))
            n_jobs_complete = len(complete_jobs)
            n_jobs_remaining = len(scheduled_jobs) - n_jobs_complete
            LOGGER.info(f"{n_jobs_complete} jobs complete. {n_jobs_remaining} jobs remaining.")
            sleep_count = 0


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

########################
### Make Predictions
########################

## Cycle Through Models
models = {}
jobs = []
for model_name, model_dir in MODEL_DIRS.items():
    LOGGER.info("Making Predictions for Model: {}".format(model_name))
    model_paths, job_ids = schedule_model_predictions(model_name, model_dir)
    models[model_name] = model_paths
    jobs.extend(job_ids)

## Hold for Complete
if not HOLD_BETWEEN:
    _ = hold_for_complete_jobs(jobs)
    LOGGER.info("Remember to remove temporary folders!")

########################
### Evaluate Model
########################

LOGGER.info("Evaluating Predictions")

## Compute Scores
scores = []
for model_name, model_set in models.items():
    for test_dataset, dataset_split in SPLITS.items():
        for fold, model in model_set.items():
            ## Load Predictions
            predictions_file = f"{OUTPUT_DIR}{model_name}/{test_dataset}/{fold}/{CONDITION}.predictions.csv"
            if not os.path.exists(predictions_file):
                LOGGER.info(f"Did not find predictions: {predictions_file}")
                continue
            predictions = pd.read_csv(predictions_file,index_col=0)
            ## Evaluate Model
            _scores = score_predictions(predictions["y_true"].values,
                                        predictions["y_pred"].values,
                                        probability=PROBABILITY,
                                        threshold=THRESHOLD)
            _scores["test_data"] = test_dataset
            _scores["train_data"] = model_name
            _scores["fold"] = int(fold.split("_")[-1])
            ## Cache
            scores.append(_scores)

## Format Scores
scores = pd.DataFrame(scores)

## Get Averages
evals = [m for m in ["precision","recall","f1","accuracy","auc"] if m in scores.columns]
avg_format = lambda x: "{:.3f} ({:.3f})".format(np.nanmean(x), np.nanstd(x)) if len(x) == 5 else "--"
scores_agg = scores.groupby(["test_data","train_data"]).agg({e:avg_format for e in evals})

## Cache Scores
scores.to_csv(f"{OUTPUT_DIR}scores.csv",index=False)
scores_agg.to_csv(f"{OUTPUT_DIR}scores_aggregated.csv",index=False)

## Display Scores
for e in evals:
    e_pivot = pd.pivot_table(scores,index="train_data",columns="test_data",values=e,aggfunc=avg_format)
    LOGGER.info("\n"+"~~~~"*5 + e + "~~~~"*5)
    LOGGER.info(e_pivot)


LOGGER.info("Script Complete!")