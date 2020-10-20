
"""
Identify distribution of posts over time for a given dataset and modeling condition
"""

#########################
### Imports
#########################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from datetime import datetime, timedelta
from collections import Counter
from multiprocessing import Pool

## External Library
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

## Local
from mhlib.util import helpers
from mhlib.util.logging import initialize_logger
from mhlib.model.train import load_dataset_metadata

#########################
### Globals
#########################

## Converters
_ = register_matplotlib_converters()

## Logger
LOGGER = initialize_logger()

## Output Paths
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)) + "/../../../"
RESULTS_DIR = f"{ROOT_PATH}data/results/cache/temporal/"

#########################
### Functions
#########################

def parse_arguments():
    """

    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Get Temporal Distribution of Posts in a Dataset")
    ## Generic Arguments
    parser.add_argument("dataset",
                        type=str,
                        choices={"clpsych",
                                 "clpsych_deduped",
                                 "multitask",
                                 "merged",
                                 "rsdd",
                                 "smhd",
                                 "wolohan"},
                        default=None,
                        help="Dataset to analyze")
    parser.add_argument("condition",
                        type=str,
                        default=None,
                        help="Which condition to isolate")
    parser.add_argument("--random_state",
                        type=int,
                        default=42,
                        help="Random sampling seed")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if args.dataset is None:
        raise ValueError("command-line argument `dataset` cannot be none")
    if args.condition is None:
        raise ValueError("command-line argument `condition` cannot be none")
    return args

def extract_timestamps(filename):
    """

    """
    with gzip.open(filename, "r") as the_file:
        data = json.load(the_file)
    timestamps = list(map(lambda x: x["created_utc"], data))
    return filename, timestamps

def convert_timestamps(ts):
    """

    """
    return list(map(lambda x: datetime.fromtimestamp(x), ts))

def load_timestamp_distribution(metadata,
                                condition):
    """

    """
    ## Isolate Filenames And Chunk for Processing
    filenames = metadata["source"].tolist()
    filename_chunks = list(helpers.chunks(filenames, 100))
    ## Label Map
    lbl_map = metadata.set_index("source")[condition].to_dict()
    ## Caches
    timestamp_distribution = {"control":Counter(),
                              condition:Counter()}
    user_time_boundaries = dict()
    ## Initialize Multiprocessing Pool
    p = Pool(8)
    ## Construct Timestamp Distribution
    for f_chunk in tqdm(filename_chunks,
                        total=len(filename_chunks),
                        position=1,
                        desc="File Chunk",
                        leave=False, file=sys.stdout):
        ## Extract Timestamps
        chunk_ts = dict(tqdm(p.imap_unordered(extract_timestamps, f_chunk),
                             total=len(f_chunk),
                             position=2,
                             desc="Filename",
                             leave=False,
                             file=sys.stdout))
        ## Convert Timestamps
        chunk_ts = dict(map(lambda x: (x[0], convert_timestamps(x[1])),
                            chunk_ts.items()))
        ## Place Timestamps in Monthly Bins
        for fname, fts in chunk_ts.items():
            f_lbl = lbl_map[fname]
            fts_counts = Counter(list(map(lambda x: (x.year, x.month), fts)))
            timestamp_distribution[f_lbl] += fts_counts
            user_time_boundaries[fname] = (min(fts), max(fts), len(fts))
    ## Close Pool
    p.close()
    return timestamp_distribution, user_time_boundaries

def format_timestamp_distribution(timestamp_distribution):
    """

    """
    ## Flatten Counts
    timestamp_distribution_df = []
    for lbl in timestamp_distribution.keys():
        dist = timestamp_distribution[lbl]
        dist = pd.DataFrame([[x,y,lbl] for x,y in dist.items()],
                             columns=["month","count","label"])
        timestamp_distribution_df.append(dist)
    timestamp_distribution_df = pd.concat(timestamp_distribution_df)
    timestamp_distribution_df["month"] = timestamp_distribution_df["month"].map(lambda i: datetime(year=i[0],month=i[1],day=1))
    timestamp_distribution_df.sort_values(["label","month"],ascending=[True,True],inplace=True)
    timestamp_distribution_df.reset_index(drop=True,inplace=True)
    return timestamp_distribution_df

def format_boundary_distribution(user_time_boundaries,
                                 metadata,
                                 condition):
    """

    """
    ## User Time Boundaries
    user_time_boundaries_df = pd.DataFrame.from_dict(user_time_boundaries,
                                                     orient="index",
                                                     columns=["min_date","max_date","num_posts"])
    user_time_boundaries_df["n_days"] = user_time_boundaries_df[["min_date","max_date"]].diff(axis=1).apply(lambda i: i["max_date"].days, axis=1)
    user_time_boundaries_df = pd.merge(user_time_boundaries_df,
                                       metadata.set_index("source")[[condition]],
                                       left_index=True,
                                       right_index=True).rename(columns={condition:"label"})
    return user_time_boundaries_df
    
def plot_timestamp_distribution(timestamp_distribution_df,
                                condition):
    """

    """
    ## Create Multiple Line Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    for l, label in enumerate(["control",condition]):
        plot_data = timestamp_distribution_df.loc[timestamp_distribution_df["label"]==label]
        plot_data = plot_data.sort_values("month", ascending=True).copy().reset_index(drop=True)
        ax.plot(plot_data["month"],
                plot_data["count"],
                label=label.title(),
                color=f"C{l}",
                linewidth=2)
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Posts Per Month", fontsize=12, fontweight="bold")
    ax.set_xlim(timestamp_distribution_df["month"].min()-timedelta(60),
                timestamp_distribution_df["month"].max()+timedelta(60))
    ax.set_yscale("symlog")
    ax.legend(loc="lower right", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax

def main():
    """

    """
    ## Parse Command Line
    args = parse_arguments()
    ## Load Dataset Metadata
    LOGGER.info("Loading Metadata")
    metadata = load_dataset_metadata(args.dataset,
                                     args.condition,
                                     random_state=args.random_state)
    ## Output Path
    LOGGER.info("Initializing Output Directory")
    outpath = f"{RESULTS_DIR}{args.dataset}-{args.condition}/"
    if not os.path.exists(outpath):
        _ = os.makedirs(outpath)
    LOGGER.info(f"Results Stored At: '{outpath}'")
    ## Extract Timestamp Distribution
    LOGGER.info("Loading Timestamp Distribution")
    timestamp_distribution, user_time_boundaries = load_timestamp_distribution(metadata,
                                                                               args.condition)
    ## Format Results
    LOGGER.info("\nFormating Distributions")
    timestamp_distribution_df = format_timestamp_distribution(timestamp_distribution)
    user_time_boundaries_df = format_boundary_distribution(user_time_boundaries,
                                                           metadata,
                                                           args.condition)
    ## Cache Results
    LOGGER.info("Caching Distributions")
    timestamp_distribution_df.to_csv(f"{outpath}timestamp_distribution.csv",index=False)
    user_time_boundaries_df.to_csv(f"{outpath}timestamp_boundaries_by_user.csv")
    ## Plot Results
    LOGGER.info("Visualizing Distributions")
    fig, ax = plot_timestamp_distribution(timestamp_distribution_df,
                                          args.condition)
    fig.savefig(f"{outpath}timestamp_distribution.png",dpi=300)
    plt.close(fig)
    ## Done
    LOGGER.info("Script complete!")

####################
### Execute
####################

if __name__ == "__main__":
    _ = main()