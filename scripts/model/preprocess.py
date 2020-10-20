

#######################
### Imports
#######################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from glob import glob
from functools import partial
from datetime import datetime

## External Library
import joblib
import pandas as pd
from tqdm import tqdm

## Local
from mhlib.util.logging import initialize_logger
from mhlib.preprocess.preprocess import (tokenizer,
                                         format_tweet_data)
from mhlib.util.multiprocessing import MyPool as Pool

#######################
### Globals
#######################

## Logging
LOGGER = initialize_logger()

## Cache Schema
DB_SCHEMA = {
    "twitter":{
        "tweet":{
                'user_id_str': 'user_id_str', 
                'created_utc': 'created_utc', 
                'text': 'text', 
                'text_tokenized': 'text_tokenized', 
                'id_str': 'tweet_id', 
                'entity_type': 'entity_type', 
                'date_processed_utc': 'date_processed_utc', 
                'source': 'source'}
    },
    "reddit":{
        "comment":{
                "author_fullname":"user_id_str",
                "created_utc":"created_utc",
                "body":"text",
                "text_tokenized":"text_tokenized",
                "id":"comment_id",
                "entity_type":"entity_type",
                'date_processed_utc': 'date_processed_utc', 
                'source': 'source'}
    }, 
}

#######################
### Functions
#######################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.

    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Preprocess raw Twitter or Reddit data")
    ## Generic Arguments
    parser.add_argument("--input",
                        type=str,
                        default=None,
                        help="Path to input folder of raw *.gz files or a single raw *.gz file")
    parser.add_argument("--output_folder",
                        type=str,
                        default=None,
                        help="Name of output folder for placing predictions.")
    parser.add_argument("--platform",
                        type=str,
                        choices=["twitter","reddit"],
                        help="Platform from which the data comes")
    parser.add_argument("--jobs",
                        type=int,
                        default=1,
                        help="Number of processes to spawn.")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if args.input is None:
        raise ValueError("Must provide --input folder or .gz file")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Could not find input filepath {args.input}")
    if args.output_folder is None:
        raise ValueError("Must provide an --output_folder argument")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    return args

def get_file_list(args):
    """

    """
    if os.path.isfile(args.input):
        return [args.input]
    elif os.path.isdir(args.input):
        return glob(f"{args.input}*.gz")
    else:
        raise ValueError("Did not recognize command line --input")

def process_tweet_file(f,
                       output_folder=None):
    """
    Process raw tweet data and cache in a processed form

    Args:
        f (str): Path to tweet data. Expected to contain all tweets
                 desired for a single individual
        output_folder (str): Path to output folder for caching processed
                             data. If None, returns processed data itself
    
    Returns:
        if output_folder is None:
            formatted_data (list): List of processed dictionaries
        else:
            fname (str): Name of file where processed data was cached
    """
    ## Load Tweet Data
    if f.endswith(".gz"):
        file_opener = gzip.open
    else:
        file_opener = open
    with file_opener(f, "r") as the_file:
        tweet_data = [format_tweet_data(i) for i in json.load(the_file)]
    ## Transform into DataFrame
    tweet_data = pd.DataFrame(tweet_data).dropna(subset=["text"])
    ## Tokenize Text
    tweet_data["text_tokenized"] = tweet_data["text"].map(tokenizer.tokenize)
    # ## Datetime Conversion
    tweet_data["created_utc"] = pd.to_datetime(tweet_data["created_at"]).map(lambda x:  int(x.timestamp()))
    ## Add Meta
    tweet_data["source"] = f
    tweet_data["entity_type"] = "tweet"
    tweet_data["date_processed_utc"] = int(datetime.utcnow().timestamp())
    ## Rename Columns and Subset
    tweet_data.rename(columns = DB_SCHEMA["twitter"]["tweet"], inplace=True)
    tweet_data = tweet_data[list(DB_SCHEMA["twitter"]["tweet"].values())]
    ## Format Into JSON
    formatted_data = tweet_data.apply(lambda row: row.to_json(), axis=1).tolist()
    formatted_data = list(map(lambda x: json.loads(x), formatted_data))
    ## Dump Processed Data (or return)
    if output_folder is None:
        return formatted_data
    else:
        fname = os.path.basename(f).replace("tweets.json","processed.tweets.json")
        if not fname.endswith(".gz"):
            fname = fname + ".gz"
        output_folder = output_folder.rstrip("/")
        fname = f"{output_folder}/{fname}"
        with gzip.open(fname, "wt", encoding="utf-8") as the_file:
            json.dump(formatted_data, the_file)
        return fname

def process_reddit_comment_file(f,
                                output_folder):
    """
    Process raw tweet data and cache in a processed form

    Args:
        f (str): Path to comment data. Expected to contain all comments
                 desired for a single individual
        output_folder (str): Path to output folder for caching processed
                             data. If None, returns processed data itself
    
    Returns:
        if output_folder is None:
            formatted_data (list): List of processed dictionaries
        else:
            fname (str): Name of file where processed data was cached
    """
    ## Load Comment Data
    if f.endswith(".gz"):
        file_opener = gzip.open
    else:
        file_opener = open
    with file_opener(f, "r") as the_file:
        comment_data = json.load(the_file)
    ## Transform into DataFrame
    comment_data = pd.DataFrame(comment_data).dropna(subset=["body"])
    ## Tokenize Text
    comment_data["text_tokenized"] = comment_data["body"].map(tokenizer.tokenize)
    ## Add Meta
    comment_data["source"] = f
    comment_data["entity_type"] = "comment"
    comment_data["date_processed_utc"] = int(datetime.utcnow().timestamp())
    ## Rename Columns and Subset
    comment_data.rename(columns = DB_SCHEMA["reddit"]["comment"], inplace=True)
    comment_data = comment_data[list(DB_SCHEMA["reddit"]["comment"].values())]
    ## Format Into JSON
    formatted_data = comment_data.apply(lambda row: row.to_json(), axis=1).tolist()
    formatted_data = list(map(lambda x: json.loads(x), formatted_data))
    ## Dump Processed Data (or return)
    if output_folder is None:
        return formatted_data
    else:
        fname = os.path.basename(f).replace("comments.json","processed.comments.json")
        if not fname.endswith(".gz"):
            fname = fname + ".gz"
        output_folder = output_folder.rstrip("/")
        fname = f"{output_folder}/{fname}"
        with gzip.open(fname, "wt", encoding="utf-8") as the_file:
            json.dump(formatted_data, the_file)
        return fname

def main():
    """

    """
    ## Parse Command-line Arguments
    args = parse_arguments()
    ## Identifty Input Files for Processing
    filenames = get_file_list(args)
    LOGGER.info("Found {} files for processing".format(len(filenames)))
    ## Identity Processor
    if args.platform == "twitter":
        mp = partial(process_tweet_file,
                     output_folder=args.output_folder)
    elif args.platform == "reddit":
        mp = partial(process_reddit_comment_file,
                     output_folder=args.output_folder)
    ## Process Files
    pool = Pool(args.jobs)
    LOGGER.info("Starting Preprocessing")
    res = list(tqdm(pool.imap_unordered(mp, filenames),
                    total=len(filenames),
                    desc="Processed Files",
                    file=sys.stdout))
    pool.close()
    LOGGER.info("Script Complete!")

#######################
### Execute
#######################

if __name__ == "__main__":
    _ = main()