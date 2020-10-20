
## Output Directories
RESULTS_DIR = "./data/results/cache/temporal/"
PLOT_DIR = "./plots/temporal/"

## Timestamp Parameters
START_DATE = "2008-01-01"
END_DATE = "2020-01-01"
FREQ = "M"
CONDITION = "depression"

## Data Sets and Balancing/Downsampling Parameters
DATA_SETS = {
        "clpsych_deduped":{
                        "downsample":False,
                        "downsample_size":500,
                        "rebalance":False,
                        "class_ratio":[1,1]
        },
        "multitask":{
                        "downsample":False,
                        "downsample_size":500,
                        "rebalance":False,
                        "class_ratio":[1,1]
        },
        "wolohan":{
                        "downsample":False,
                        "downsample_size":500,
                        "rebalance":True,
                        "class_ratio":[1,1]
        },
        "rsdd":{
                        "downsample":False,
                        "downsample_size":500,
                        "rebalance":True,
                        "class_ratio":[1,1]
        },
        "smhd":{
                        "downsample":False,
                        "downsample_size":500,
                        "rebalance":True,
                        "class_ratio":[1,1]
        },
}

## Meta Parameters
RANDOM_SEED = 42
NUM_PROCESSES = 8

#######################
### Imports
#######################

## Standard Library
import os
import sys
import gzip
import json
from datetime import datetime
from collections import Counter
from multiprocessing import Pool

## External
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack

## Local
from mhlib.model import train
from mhlib.model.data_loaders import LoadProcessedData
from mhlib.util.logging import initialize_logger

#######################
### Globals
#######################

## Logger
LOGGER = initialize_logger()

## Data Loader
DATA_LOADER = LoadProcessedData(filter_mh_subreddits="all",
                                filter_mh_terms="smhd",
                                max_documents_per_user=None)

## Timestamp Bins
DATE_RANGE_BINS = pd.date_range(START_DATE, END_DATE, freq=FREQ)

## Directories
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

#######################
### Helpers
#######################

def unify_metadata(df,
                   dataset):
    """
    Isolate unified set of columns across data set metadata files

    Args:
        df (pandas DataFrame): Metadata DataFrame
        dataset (set): Name of the dataset
    
    Returns:
        df (pandas DataFrame): Formatted Metadata DataFrame
    """
    ## Rename Post Columns
    col_mapping = {
        "num_tweets":"num_posts",
        "num_comments":"num_posts"
    }
    for c, cnew in col_mapping.items():
        df = df.rename(columns={c:cnew})
    ## Add Dataset Flag
    df["dataset"] = dataset
    ## Generalize Split if Missing
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].fillna("train")
    ## Subset to Relevant Metadata Columns
    col_subset = ["user_id_str",CONDITION,"gender","age","split","num_posts","num_words","dataset","source"]
    df = df[[c for c in col_subset if c in df.columns]].copy()
    return df

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

def load_timestamp_vector(filename):
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
    user_data = DATA_LOADER.load_user_data(filename)
    ## No Data after Filtering
    if len(user_data) == 0:
        return filename, csr_matrix(np.zeros(len(DATE_RANGE_BINS)))
    ## Get Timestamps
    timestamps = list(map(lambda u: datetime.fromtimestamp(u["created_utc"]), user_data))
    ## Assign To Bins
    timestamp_bins = assign_date_values_to_bins(timestamps, DATE_RANGE_BINS)
    ## Filter Out Null Bins
    timestamp_bins = list(filter(lambda tb: tb is not None, timestamp_bins))
    ## Count Distribution
    timestamp_count = Counter(timestamp_bins)
    ## Create Vector
    x = np.zeros(len(DATE_RANGE_BINS))
    for tcb, tcc in timestamp_count.items():
        x[tcb] = tcc
    x = csr_matrix(x)
    return filename, x
    
def plot_statistic_heatmap(statistics,
                           metric="users_greater_than_0_posts"):
    """

    """
    ## Pivot Table
    stats_pivot = pd.pivot_table(statistics,
                                 index=["dataset",CONDITION],
                                 columns=["year"],
                                 values=[metric],
                                 aggfunc=max)[metric]
    stats_pivot_normed = stats_pivot.apply(lambda x: x / max(x), axis=1)
    ## Heat Map
    fig, ax = plt.subplots(figsize=(10,5.8))
    m = ax.imshow(stats_pivot_normed,
                  interpolation="nearest",
                  cmap=plt.cm.Purples,
                  aspect="auto",
                  alpha=1)
    ax.set_xticks(list(range(stats_pivot.shape[1])))
    ax.set_xticklabels(stats_pivot.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(list(range(stats_pivot.shape[0])))
    ax.set_yticklabels(["{} ({})".format(x,y) for x,y in stats_pivot.index],
                       fontsize=6)
    ax.set_ylim(stats_pivot.shape[0]-.5, -.5)
    for i, row in enumerate(stats_pivot.values):
        for j, val in enumerate(row):
            if val > 0:
                ax.text(j, i, int(val),
                        ha="center",
                        va="center",
                        color = "white" if stats_pivot_normed.values[i,j] > .5 else "black",
                        fontsize=6)
    ax.set_title(metric, loc = "left", fontsize=10)
    fig.tight_layout()
    return fig, ax

#######################
### Load Metadata
#######################

LOGGER.info("Loading Dataset Metadata")

## Load Datasets Jointly
datasets_df = []
for ds in DATA_SETS:
    ## Load Metadata
    ds_df = train.load_dataset_metadata(ds,
                                        CONDITION,
                                        RANDOM_SEED)
    ## Balance/Downsample
    if DATA_SETS[ds]["rebalance"]:
        ds_df = train._rebalance(ds_df,
                                 CONDITION,
                                 DATA_SETS[ds]["class_ratio"],
                                 RANDOM_SEED)
    if DATA_SETS[ds]["downsample"]:
        ds_df = train._downsample(ds_df,
                                  DATA_SETS[ds]["downsample_size"],
                                  RANDOM_SEED
                )
    ## Unify Columns
    ds_df = unify_metadata(ds_df, ds)
    datasets_df.append(ds_df)
datasets_df = pd.concat(datasets_df, sort=False)

#######################
### Load Timestamp Vectors
#######################

LOGGER.info("Loading Timestamp Distribution")

## Load Vectors
mp = Pool(NUM_PROCESSES)
res = list(tqdm(mp.imap_unordered(load_timestamp_vector,
                                  datasets_df["source"].tolist()),
                file=sys.stdout,
                total=len(datasets_df),
                ))
mp.close()

## Parse Vectors
filenames = [r[0] for r in res]
X = vstack([r[1] for r in res])

## Map Filenames to Metadata
dataset_membership = np.array(datasets_df.set_index("source").loc[filenames]["dataset"].tolist())
condition_labels = np.array(datasets_df.set_index("source").loc[filenames][CONDITION].tolist())

## Date Bin Mapping
date_boundaries = []
for x, y in zip(DATE_RANGE_BINS[:-1], DATE_RANGE_BINS[1:]):
    date_boundaries.append((x.date(),y.date()))
date_boundaries.append((y.date(), datetime.now().date()))

## Cache Raw Data
LOGGER.info("Caching Raw Data")
_ = joblib.dump({"X":X,
                 "filenames":filenames,
                 "dataset_membership":dataset_membership,
                 "condition_labels":condition_labels,
                 "date_range_bins":DATE_RANGE_BINS,
                 "date_boundaries":date_boundaries},
                f"{RESULTS_DIR}timestamp_distribution.joblib")

#######################
### Analysis
#######################

LOGGER.info("Computing Summary Statistics")

## Year Boundaries
year_bounds = list(range(date_boundaries[0][0].year,
                         date_boundaries[-1][1].year + 1))

## Cycle Through Splits (Record Statistics)
statistics = {}
for dataset in tqdm(DATA_SETS, position=0,leave=False, file=sys.stdout):
    for group in tqdm([CONDITION,"control"], position=1, leave=False, file=sys.stdout):
        ## Dataset, Label Subset
        col_mask = np.logical_and(condition_labels == group, dataset_membership == dataset)
        for year_start, year_end in tqdm(list(zip(year_bounds[:-1],year_bounds[1:])), position = 2, leave=False, file=sys.stdout):
            ## Time-based Subset
            year_mask = [i[0].year >= year_start and i[1].year < year_end for i in date_boundaries]
            ## Isolate Post Data
            X_subset = X[np.nonzero(col_mask)[0]][:, np.nonzero(year_mask)[0]]
            ## Sum over Users
            posts_per_user = np.array(X_subset.sum(axis=1).T)[0]
            ## Get Statistics
            split_stats = {
                          "min_posts":posts_per_user.min(),
                          "max_posts":posts_per_user.max(),
                          "mean_posts":posts_per_user.mean(),
                          "median_posts":np.median(posts_per_user)
            }
            for threshold in [0, 20, 40, 60, 80, 100, 200, 300]:
                split_stats[f"users_greater_than_{threshold}_posts"] = (posts_per_user > threshold).sum()
            ## Record
            statistics[(dataset, group, year_start)] = split_stats

## Format into DataFrame
statistics = pd.DataFrame.from_dict(statistics, orient="index")
statistics = statistics.reset_index().rename(columns={"level_0":"dataset",
                                                      "level_1":CONDITION,
                                                      "level_2":"year"})

## Cache Statistics
LOGGER.info("Caching Summary Statistics")
statistics.to_csv(f"{RESULTS_DIR}post_statistics_by_year.csv", index=False)

#######################
### Visualization
#######################

LOGGER.info("Creating Summary Statistic Visualizations")

## Plot Statistics
for metric in statistics.drop(["dataset",CONDITION,"year","min_posts"],axis=1).columns:
    fig, ax = plot_statistic_heatmap(statistics, metric)
    fig.savefig(f"{PLOT_DIR}{metric}.png", dpi=150)
    plt.close(fig)

LOGGER.info("Script Complete.")
