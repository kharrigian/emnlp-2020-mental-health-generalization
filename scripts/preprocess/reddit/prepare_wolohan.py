
## Multiprocessing
NUM_PROCESSES = 8

## Data Directory Path and Dataset Prefix
RAW_DATA_DIR = "./data/raw/reddit/wolohan/"
PROCESSED_DATA_DIR = "./data/processed/reddit/wolohan/"
LABEL_FILE = f"{RAW_DATA_DIR}wolohan_label_map.joblib" # Output from scripts/acquire/get_wolohan.py

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

## External Libaries
import joblib
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

## Depression/Control Labels
author_labels = joblib.load(LABEL_FILE)

#####################
### Helper Functions
#####################

def process_comment_df(comment_df,
                       current_date,
                       source):
    """
    Process a comment data chunk

    Args:
        comment_df (pandas DataFrame): Raw cached comment dataframe
        current_date (int): UTC timestamp
        source (str): Source filepath
    
    Returns:
        None, saves processed data and metadata
    """
    ## Add Label
    comment_df["depression"] = comment_df["author"].map(lambda i: author_labels[i])
    ## Tokenize the text
    comment_df["text_tokenized"] = comment_df["body"].map(tokenizer.tokenize)
    ## Link ID
    comment_df["link_id"] = comment_df["link_id"].map(lambda i: i.split("_")[1])
    ## Entity Type
    comment_df["entity_type"] = "comment"
    ## Dataset
    comment_df["dataset"] = comment_df.index.map(lambda i: ["wolohan"])
    ## Data Source
    comment_df["source"] = os.path.abspath(source)
    ## Processing Date
    comment_df["date_processed_utc"] = current_date
    ## Author Groups
    author_groups = comment_df.groupby(["author"]).groups
    author_groups = dict((x, list(y)) for x,y in author_groups.items())
    ## Rename Columns and Subset
    comment_df.rename(columns = db_schema["wolohan"]["comment"], inplace=True)
    comment_df = comment_df[list(db_schema["wolohan"]["comment"].values())]
    ## Format Into JSON
    formatted_data = comment_df.apply(lambda row: row.to_json(), axis=1).tolist()
    formatted_data = list(map(lambda x: json.loads(x), formatted_data))
    ## Dump
    for author, author_index in author_groups.items():
        ## Get Author-specific Data
        author_formatted_data = [formatted_data[i] for i in author_index]
        ## Dump Processed Comments
        comment_outfile = f"{PROCESSED_DATA_DIR}{author}.comments.tar.gz"
        with gzip.open(comment_outfile, "wt", encoding="utf-8") as the_file:
            json.dump(author_formatted_data, the_file)
        ## Dump Metadata
        meta_outfile = f"{PROCESSED_DATA_DIR}{author}.comments.meta.tar.gz"
        meta_dict = {"user_id_str":author,
                     "depression":author_labels[author],
                     "datasets":["wolohan"],
                     "num_comments":len(author_formatted_data),
                     "num_words": sum([len(i["text_tokenized"]) for i in author_formatted_data])
        }
        with gzip.open(meta_outfile, "wt", encoding="utf-8") as the_file:
            json.dump(meta_dict, the_file)

def process_submission_df(submission_df,
                          current_date,
                          source):
    """
    Process a submission data chunk

    Args:
        submission_df (pandas DataFrame): Raw cached submission dataframe
        current_date (int): UTC timestamp
        source (str): Source filepath
    
    Returns:
        None, saves processed data and metadata
    """
    ## Add Label
    submission_df["depression"] = submission_df["author"].map(lambda i: author_labels[i])
    ## Tokenize the text
    submission_df["text_tokenized"] = submission_df["selftext"].map(tokenizer.tokenize)
    submission_df["title_tokenized"] = submission_df["title"].map(tokenizer.tokenize)
    ## Entity Type
    submission_df["entity_type"] = "submission"
    ## Data Source
    submission_df["source"] = source
    ## Dataset
    submission_df["dataset"] = submission_df.index.map(lambda i: ["wolohan"])
    ## Processing Date
    submission_df["date_processed_utc"] = current_date
    ## Get Author Groupings
    author_groups = submission_df.groupby(["author"]).groups
    author_groups = dict((x, list(y)) for x,y in author_groups.items())
    ## Rename Columns and Subset
    submission_df.rename(columns = db_schema["wolohan"]["submission"], inplace=True)
    submission_df = submission_df[list(db_schema["wolohan"]["submission"].values())]
    ## Format Into JSON
    formatted_data = submission_df.apply(lambda row: row.to_json(), axis=1).tolist()
    formatted_data = list(map(lambda x: json.loads(x), formatted_data))
    ## Dump
    for author, author_index in author_groups.items():
        ## Get Author-specific Data
        author_formatted_data = [formatted_data[i] for i in author_index]
        ## Dump Processed Comments
        submission_outfile = f"{PROCESSED_DATA_DIR}{author}.submissions.tar.gz"
        with gzip.open(submission_outfile, "wt", encoding="utf-8") as the_file:
            json.dump(author_formatted_data, the_file)
        ## Dump Meta
        meta_outfile = f"{PROCESSED_DATA_DIR}{author}.submissions.meta.tar.gz"
        meta_dict = {"user_id_str":author,
                     "depression":author_labels[author],
                     "datasets":["wolohan"],
                     "num_submissions":len(author_formatted_data),
                     "num_words": sum([len(i["text_tokenized"]) for i in author_formatted_data])
        }
        with gzip.open(meta_outfile, "wt", encoding="utf-8") as the_file:
            json.dump(meta_dict, the_file)

def process_file(f,
                 current_date):
    """
    Process a reddit data chunk

    Args:
        f (str): Input filepath of the data chunk
        current_date (int): UTC timestamp
    
    Returns:
        None, saves processed data and metadata
    """
    ## Submission or Comments
    is_submission = os.path.basename(f).startswith("submission")
    ## Load Data File
    f_data = joblib.load(f).reset_index(drop=True).copy()
    ## Continue if No Records
    if len(f_data) == 0:
        return
    ## Process Data
    if is_submission:
        formatted_data = process_submission_df(f_data,
                                               current_date,
                                               os.path.abspath(f))
    else:
        formatted_data = process_comment_df(f_data,
                                            current_date,
                                            os.path.abspath(f))
    ## Logging
    logger.info(f"Successfully processed {f}")
    return

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
    ## Get Raw Data Files (Comment Histories)
    data_files = []
    for s, subreddit in enumerate(["depression","AskReddit"]):
        s_files = glob(f"{RAW_DATA_DIR}wolohan_{subreddit}_histories/*")
        data_files.extend(s_files)
    ## Create Multiprocessing Pool and Execute
    mp_pool = Pool(processes=NUM_PROCESSES)
    helper = partial(process_file, current_date=current_date)
    _ = mp_pool.map(helper, data_files)
    mp_pool.close()

#####################
### Execution
#####################

if __name__ == "__main__":
    _ = main()
