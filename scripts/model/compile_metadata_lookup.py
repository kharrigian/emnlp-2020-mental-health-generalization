
"""
Create Metadata CSVs for easy file lookup during modeling. Expects
that all prepare_*.py scripts have been run already.
"""

## Enumerate Datasets
DATASETS = ["clpsych",
            "clpsych_deduped",
            "multitask",
            "merged",
            "wolohan",
            "rsdd",
            "smhd"]

## Choose Number of Cores to Use
NUM_PROCESSES = 8

#################
### Imports
#################

## Standard Libary
import os
import sys
import json
import gzip
from glob import glob
from multiprocessing import Pool

## External Libaries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

## Local Modules
from mhlib.util.logging import initialize_logger
from mhlib.model.data_loaders import LoadProcessedData
from mhlib.model.train import DATA_PATHS

## Create Logger
logger = initialize_logger()

#################
### Dictionaries/Global Variables
#################

## Initialize Data Loader
LOADER = LoadProcessedData()

#################
### Functions
#################

def _metadata_helper(f):
    """
    Helper for loading user metadata with 
    associated filename.

    Args:
        f (str): Path to user metadata file
    
    Returns;
        (f, user_metdata): (str, dict): Filename and user metadata
    """ 
    return (f, LOADER.load_user_metadata(f))

def load_dataset_metadata(dataset,
                          jobs=8):
    """
    Load metadata for a particular data set

    Args:
        dataset (str): Canonical name of the dataset
        jobs (int): Number of multiprocessing processes to use
    
    Returns:
        metadata_df (pandas DataFrame): Metadata dataframe, sorted
                        by the "user_id_str" in ascending order.
    """
    ## Get Key
    dpath_key = [d for d in DATA_PATHS.keys() if dataset in d][0]
    dpath = DATA_PATHS[dpath_key]
    ## Load Metadata
    meta_files = glob("{}*{}".format(dpath["path"], dpath["meta_suffix"]))
    mp = Pool(processes=jobs)
    metadata = list(tqdm(mp.imap_unordered(_metadata_helper, meta_files),
                         total=len(meta_files),
                         desc="Loading Metadata",
                         file=sys.stdout))
    mp.close()
    ## Format Metadata
    metadata_df = pd.DataFrame([m[1] for m in metadata])
    ## Add Source and Check For File Existences
    metadata_df["source"] = [m[0] for m in metadata]
    metadata_df["source"] = metadata_df["source"].str.replace(".meta","")
    metadata_df = metadata_df.loc[metadata_df.source.map(os.path.exists)]
    metadata_df["source"] = metadata_df["source"].map(os.path.abspath)
    ## Isolate Proper Dataset
    metadata_df = metadata_df.loc[metadata_df["datasets"].map(lambda i: dataset.split("-")[0] in i)]
    ## Reset Index and Copy
    metadata_df = metadata_df.sort_values(["user_id_str"], ascending=True)
    metadata_df = metadata_df.reset_index(drop=True).copy()
    return metadata_df

def cache_metadata(metadata_df,
                   dataset):
    """
    Save the Metadata DataFrame to Disk

    Args:
        metadata_df (pandas DataFrame): Metadata for the dataset. Output from
                                        load_dataset_metadata().
        dataset (str): Name of the dataset.
    
    Returns:
        None, saves dataset to disk.
    """
    ## Get Key
    dpath_key = [d for d in DATA_PATHS.keys() if dataset in d][0]
    dpath = DATA_PATHS[dpath_key]
    ## Get Output path
    outpath = "{}{}_metadata.csv".format(dpath["path"], dataset)
    _ = metadata_df.to_csv(outpath, index=False)
    logger.info(f"Saved {dataset} to {outpath}")

def process_dataset(dataset):
    """
    Load metadata for a dataset and cache it to disk.

    Args:
        dataset (str): Canonical name of a dataset
    
    Returns:
        None
    """
    ## Load Metadata into DataFrame
    metadata_df = load_dataset_metadata(dataset,
                                        jobs=NUM_PROCESSES)
    ## Cache Metadata
    _ = cache_metadata(metadata_df,
                       dataset)

def main():
    """
    Process all mental-health datasets, caching their
    metadata to disk for easy access.
    """
    ## Process Each Dataset in Serial
    for d, dataset in enumerate(DATASETS):
        logger.info(f"Processing dataset {d+1}/{len(DATASETS)}: `{dataset}`")
        _ = process_dataset(dataset)

#################
### Execute
#################

if __name__ == "__main__":
    _ = main()