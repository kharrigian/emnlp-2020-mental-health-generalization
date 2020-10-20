
## Multiprocessing
NUM_PROCESSES = 8

## Input Data Directories
LABEL_FILE="./data/raw/twitter/qntfy/qntfy_merged_labels.csv"

## Output Data Directories
PROCESSED_DATA_DIR = "./data/processed/twitter/qntfy/"

## Flag to remove existing data
REMOVE_EXISTING = True

#####################
### Imports
#####################

## Standard Libraries
import os
import json
import gzip
from glob import glob
from functools import partial
from datetime import datetime
from multiprocessing import Pool

## External Libraries
import pandas as pd

## Local Modules
from mhlib.util.logging import initialize_logger
from mhlib.preprocess.preprocess import (tokenizer,
                                         format_tweet_data,
                                         db_schema)

#####################
### Globals
#####################

## Create Logger
logger = initialize_logger()

## Load Labels
author_labels = pd.read_csv(LABEL_FILE)
author_labels.set_index("source", inplace=True)

#####################
### Functions
#####################

def process_tweet_file(f,
                       current_date):
    """
    Process raw tweet data in the QNTFY data sets and 
    cache in a processed form along with user metadata

    Args:
        f (str): Path to tweet data, relative to base QNTFY path
        current_date (int): UTC datetime in seconds
    
    Returns:
        None
    """
    ## Parse Author
    author_meta = author_labels.loc[f].to_dict()
    author = author_meta["user_id_str"]
    ## Identify Data Sets This User Should Match Into
    datasets_allowed = []
    for ds in ["clpsych","clpsych_deduped","multitask","merged"]:
        if author_meta[ds]:
            datasets_allowed.append(ds)
    ## Output Files
    tweet_outfile = f"{PROCESSED_DATA_DIR}{author}.tweets.tar.gz"
    meta_outfile = f"{PROCESSED_DATA_DIR}{author}.tweets.meta.tar.gz"
    if os.path.exists(tweet_outfile) and os.path.exists(meta_outfile):
        logger.info(f"Already processed data for user: {author}")
        return
    logger.info(f"Processing data for user: {author}")
    ## Load Tweet Data
    tweet_data = []
    if f.endswith(".gz"):
        file_opener = gzip.open
    else:
        file_opener = open
    with file_opener(f, "r") as the_file:
        for line in the_file:
            data = json.loads(line)
            tweet_data.append(format_tweet_data(data))
    ## Transform into DataFrame
    tweet_data = pd.DataFrame(tweet_data)
    tweet_data["user_id_str"] = author
    ## Tokenize Text
    tweet_data["text_tokenized"] = tweet_data["text"].map(tokenizer.tokenize)
    ## Initialize Metadata Dictionary
    meta_dict = {"user_id_str":author,
                 "datasets":datasets_allowed,
                 "age":author_meta["age"] if not pd.isnull(author_meta["age"]) else None,
                 "gender":author_meta["gender"] if not pd.isnull(author_meta["gender"]) else None,
                 "split":author_meta["split"] if not pd.isnull(author_meta["split"]) else None,
                 "user_id_str_matched":author_meta["user_id_str_matched"],
                 "num_tweets":int(len(tweet_data)),
                 "num_words": int(tweet_data.text_tokenized.map(len).sum())
    }
    ## Datetime Conversion
    tweet_data["created_utc"] = pd.to_datetime(tweet_data["created_at"]).map(lambda x:  int(x.timestamp()))
    ## Add Labels
    is_control = False
    if author_meta["neurotypical"]:
        is_control = True
    for disorder in ["anxiety",
                     "depression",
                     "suicide_attempt",
                     "suicidal_ideation",
                     "eating_disorder",
                     "panic",
                     "schizophrenia",
                     "borderline",
                     "bipolar",
                     "ptsd"]:
        if is_control:
            tweet_data[disorder] = "control"
            meta_dict[disorder] = "control"
        else:
            if author_meta[disorder]:
                tweet_data[disorder] = disorder
                meta_dict[disorder] = disorder
    ## Add Meta
    tweet_data["source"] = os.path.abspath(f)
    tweet_data["entity_type"] = "tweet"
    tweet_data["date_processed_utc"] = current_date
    ## Add Author Metadata
    tweet_data["age"] = author_meta["age"] if not pd.isnull(author_meta["age"]) else None
    tweet_data["gender"] = author_meta["gender"] if not pd.isnull(author_meta["gender"]) else None
    ## Add Dataset Indicators
    tweet_data["dataset"] = tweet_data.index.map(lambda i: datasets_allowed)
    ## Rename Columns and Subset
    rel_cols = dict((x, y) for x, y in db_schema["qntfy"]["tweet"].items() if x in tweet_data.columns)
    tweet_data.rename(columns = rel_cols, inplace=True)
    tweet_data = tweet_data[list(rel_cols.values())]
    ## Format Into JSON
    formatted_data = tweet_data.apply(lambda row: row.to_json(), axis=1).tolist()
    formatted_data = list(map(lambda x: json.loads(x), formatted_data))
    ## Dump Processed Comments
    with gzip.open(tweet_outfile, "wt", encoding="utf-8") as the_file:
        json.dump(formatted_data, the_file)
    ## Dump Author Metadata
    with gzip.open(meta_outfile, "wt", encoding="utf-8") as the_file:
        json.dump(meta_dict, the_file)
    ## Alert User Data is Processed
    logger.info(f"Successfully processed user: {f}")
    return

def main():
    """

    """
    ## Setup Processed Data Directory
    if os.path.exists(PROCESSED_DATA_DIR) and REMOVE_EXISTING:
        logger.info(f"Removing any existing preprocessed data from {PROCESSED_DATA_DIR}")
        _ = os.system(f"rm -rf {PROCESSED_DATA_DIR}")
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    ## Get Data Files
    data_files =  author_labels.index.tolist()
    ## Processing Time
    current_date = int(datetime.utcnow().timestamp())
    ## Create Multiprocessing Pool and Execute
    mp_pool = Pool(processes=NUM_PROCESSES)
    helper = partial(process_tweet_file, current_date=current_date)
    _ = mp_pool.map(helper, data_files)
    mp_pool.close()
    ## Script Complete
    logger.info("Preprocessing Complete.")

#####################
### Execution
#####################

if __name__ == "__main__":
    _ = main()