
"""
Prepare a tokenized version of the RSDD dataset.
"""

## Multiprocessing
NUM_PROCESSES = 8

## Data Directory Paths
RAW_DATA_DIR = "./data/raw/reddit/RSDD/"
PROCESSED_DATA_DIR = "./data/processed/reddit/rsdd/"

## Flag to remove existing data
REMOVE_EXISTING = True

#####################
### Imports
#####################

## Standard Libraries
import os
import json
import gzip
from functools import partial
from datetime import datetime
from multiprocessing import Pool

## External Libraries
import pandas as pd

## Local Modules
from mhlib.util.logging import initialize_logger
from mhlib.preprocess.preprocess import (tokenizer,
                                         db_schema)

#####################
### Globals
#####################

## Create Logger
logger = initialize_logger()

#####################
### Helper Functions
#####################

def process_json_line(line,
                      filename,
                      current_date):
    """
    Process a line of data from the gzipped RSDD file.

    Args:
        params (dict): Contains parameters passed through the
                       process_file function + a line of the
                       uncompressed fine
    
    Returns:
        None, saves tokenized data to disk
    """
    ## Identify Split
    split = {"training.gz":"train",
             "validation.gz":"dev",
             "testing.gz":"test"}[filename.split("/")[-1]]
    ## Load Line as a JSON
    data = json.loads(line)[0]
    ## Output Files
    author = data["id"]
    posts_outfile = f"{PROCESSED_DATA_DIR}{author}.posts.tar.gz"
    meta_outfile = f"{PROCESSED_DATA_DIR}{author}.posts.meta.tar.gz"
    logger.info(f"Starting to process User: {author}")
    if os.path.exists(posts_outfile) and os.path.exists(meta_outfile):
        logger.info(f"Already Processed User: {author}")
        return
    ## Create DataFrame
    text_df = pd.DataFrame(data["posts"], columns = ["created_utc","text"])
    ## Add Metadata
    text_df["entity_type"] = "post"
    text_df["date_processed_utc"] = current_date
    text_df["depression"] = data["label"]
    text_df["user_id_str"] = data["id"]
    text_df["dataset"] = text_df.index.map(lambda i: ["rsdd"])
    text_df["source"] = os.path.abspath(filename)
    ## Tokenize Text
    text_df["text_tokenized"] = text_df["text"].map(tokenizer.tokenize)
    ## Rename Columns and Subset
    text_df.rename(columns = db_schema["rsdd"]["comment"], inplace=True)
    text_df = text_df[list(db_schema["rsdd"]["comment"].values())]
    ## Format Into JSON
    formatted_data = text_df.apply(lambda row: row.to_json(), axis=1).tolist()
    formatted_data = list(map(lambda x: json.loads(x), formatted_data))
    ## Dump Post Data
    with gzip.open(posts_outfile, "wt", encoding="utf-8") as the_file:
        json.dump(formatted_data, the_file)
    ## Dump Metadata
    meta_dict = {"user_id_str":author,
                 "depression":data["label"],
                 "datasets":["rsdd"],
                 "split":split,
                 "num_comments":len(formatted_data),
                 "num_words": sum([len(i["text_tokenized"]) for i in formatted_data])
    }
    with gzip.open(meta_outfile, "wt", encoding="utf-8") as the_file:
        json.dump(meta_dict, the_file)
    logger.info("Processed User:{}".format(author))
    return

def process_file(filename,
                 current_date):
    """
    Process a file from the RSDD file

    Args:
        filename (str): Name of the input .gz file
        current_date (int): UTC epoch time
    
    Returns:
        None
    """
    ## Initialize Pool
    mp = Pool(NUM_PROCESSES)
    helper = partial(process_json_line, filename=filename, current_date=current_date)
    ## Read File and Apply Processing Function
    with gzip.open(filename, "r") as the_file:
        _ = mp.map(helper, the_file)
    ## Close Pool
    mp.close()

def main():
    """

    """
    ## Setup Processed Data Directory
    if os.path.exists(PROCESSED_DATA_DIR) and REMOVE_EXISTING:
        _ = os.system(f"rm -rf {PROCESSED_DATA_DIR}")
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    ## Script Metadata
    current_date = int(datetime.utcnow().timestamp())
    ## Data Files
    data_files = [f"{RAW_DATA_DIR}{s}.gz" for s in ["training","validation","testing"]]
    ## Process Files
    for df in data_files:
        logger.info(f"Processing File: {df}")
        _ = process_file(df,
                         current_date)

#####################
### Execute
#####################

if __name__ == "__main__":
    _ = main()