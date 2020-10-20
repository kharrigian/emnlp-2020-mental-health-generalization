
"""
Prepare a tokenized version of the SMHD dataset. Only processes
users matched as "control" or "depression"
"""

## Multiprocessing
NUM_PROCESSES = 8

## Data Directory Paths
RAW_DATA_DIR = "./data/raw/reddit/SMHDv1.1/"
PROCESSED_DATA_DIR = "./data/processed/reddit/smhd/"

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
        line (bytes): JSON string
        filename (str): Which file the data comes from
        current_date (int): Processing time
    
    Returns:
        None, saves tokenized data to disk
    """
    ## Load Line as a JSON
    data = json.loads(line)
    ## Identify Split
    split = {"SMHD_train.jl.gz":"train",
             "SMHD_dev.jl.gz":"dev",
             "SMHD_test.jl.gz":"test",
             "RC.jl.gz":"relaxed"}[filename.split("/")[-1]]
    ## Labeling
    is_control = False
    if "control" in data["label"]:
        is_control = True
    labels = {}
    for disorder in ["depression",
                     "adhd",
                     "anxiety",
                     "autism",
                     "bipolar",
                     "eating",
                     "ocd",
                     "ptsd",
                     "schizophrenia"]:
        disorder_label = db_schema["smhd"]["comment"][disorder]
        if is_control:
            labels[disorder_label] = "control"
        else:
            if disorder in data["label"]:
                labels[disorder_label] = disorder_label
            else:
                labels[disorder_label] = None
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
    text_df["user_id_str"] = data["id"]
    text_df["dataset"] = text_df.index.map(lambda i: ["smhd"])
    text_df["source"] = os.path.abspath(filename)
    ## Add Labels
    for disorder, status in labels.items():
        text_df[disorder] = status
    ## Tokenize Text
    text_df["text_tokenized"] = text_df["text"].map(tokenizer.tokenize)
    ## Rename Columns and Subset
    text_df.rename(columns = db_schema["smhd"]["comment"], inplace=True)
    text_df = text_df[list(db_schema["smhd"]["comment"].values())]
    ## Format Into JSON
    formatted_data = text_df.apply(lambda row: row.to_json(), axis=1).tolist()
    formatted_data = list(map(lambda x: json.loads(x), formatted_data))
    ## Dump Post Data
    with gzip.open(posts_outfile, "wt", encoding="utf-8") as the_file:
        json.dump(formatted_data, the_file)
    ## Dump Metadata
    meta_dict = {"user_id_str":author,
                 "datasets":["smhd"],
                 "split":split,
                 "num_comments":len(formatted_data),
                 "num_words": sum([len(i["text_tokenized"]) for i in formatted_data])
    }
    for disorder, status in labels.items():
        meta_dict[disorder] = status
    with gzip.open(meta_outfile, "wt", encoding="utf-8") as the_file:
        json.dump(meta_dict, the_file)
    logger.info("Processed User:{}".format(author))
    return

def process_file(filename,
                 current_date):
    """
    Process a file from the SMHD file

    Args:
        filename (str): Name of the input .gz file
        current_date (int): UTC epoch time
    
    Returns:
        None
    """
    ## Initialize Pool
    mp = Pool(processes=NUM_PROCESSES)
    ## Helper Function with Parameters
    mp_func = partial(process_json_line,
                      filename=filename,
                      current_date=current_date)
    ## Read File and Map Processing Function
    with gzip.open(filename, "r") as the_file:
        _ = list(mp.imap_unordered(mp_func, the_file))
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
    data_files = [f"{RAW_DATA_DIR}SMHD_{s}.jl.gz" for s in ["train","dev","test"]]
    data_files.append(f"{RAW_DATA_DIR}RC.jl.gz")
    ## Process Files
    for df in data_files:
        logger.info(f"Processing File: {df}")
        _ = process_file(df,
                         current_date)

#####################
### Execution
#####################

if __name__ == "__main__":
    _ = main()