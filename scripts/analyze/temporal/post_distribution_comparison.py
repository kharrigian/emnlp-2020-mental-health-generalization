
## Directories
PLOT_DIR = "./plots/temporal/"
DATA_DIR = "./data/processed/"
RESULTS_DIR = "./data/results/cache/temporal/"

## Condition
CONDITION = "depression"

## Data Sets and Balancing/Downsampling Parameters
DATA_SETS = {
        "clpsych_deduped":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":False,
                        "class_ratio":[1,1]
        },
        "multitask":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":False,
                        "class_ratio":[1,1]
        },
        "wolohan":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":True,
                        "class_ratio":[1,1]
        },
        "rsdd":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":True,
                        "class_ratio":[1,1]
        },
        "smhd":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":True,
                        "class_ratio":[1,1]
        },
}

## Random Seed
RANDOM_SEED = 42

########################
### Imports
########################

## Standard Library
import os
import gzip
import json
import sys
from datetime import datetime, timedelta
from collections import Counter
from multiprocessing.dummy import Pool

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

## Local
from mhlib.model import train
from mhlib.util.helpers import flatten, chunks
from mhlib.util.logging import initialize_logger

## Register Timestamp Converters
register_matplotlib_converters()

########################
### Helpers
########################

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

########################
### Globals
########################

## Initialize Logger
logger = initialize_logger()

## Initialize Cache
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

## Initialize Plot Directory
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

########################
### Load Metadata
########################

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

########################
### Extract Timestamps
########################

## Isolate Filenames And Chunk for Processing
filenames = datasets_df["source"].tolist()
filename_chunks = list(chunks(filenames, 100))

## Label and Dataset Map
ds_map = datasets_df.set_index("source")["dataset"].to_dict()
lbl_map = datasets_df.set_index("source")[CONDITION].to_dict()

## Caches
timestamp_distribution = dict((ds, {"control":Counter(),CONDITION:Counter()}) for ds in DATA_SETS)
user_time_boundaries = dict()

## Construct Timestamp Distribution
p = Pool(8)
for f_chunk in tqdm(filename_chunks, total=len(filename_chunks), position=1, desc="File Chunk", leave=False, file=sys.stdout):
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
        f_ds = ds_map[fname]
        f_lbl = lbl_map[fname]
        fts_counts = Counter(list(map(lambda x: (x.year, x.month), fts)))
        timestamp_distribution[f_ds][f_lbl] += fts_counts
        user_time_boundaries[fname] = (min(fts), max(fts), len(fts))
p.close()

########################
### Analysis
########################

## Flatten Counts
timestamp_distribution_df = []
for dataset in timestamp_distribution.keys():
    for lbl in [CONDITION,"control"]:
        dist = timestamp_distribution[dataset][lbl]
        dist = pd.DataFrame([[x,y,dataset,lbl] for x,y in dist.items()],
                            columns=["month","count","dataset",CONDITION])
        timestamp_distribution_df.append(dist)
timestamp_distribution_df = pd.concat(timestamp_distribution_df)
timestamp_distribution_df["month"] = timestamp_distribution_df["month"].map(lambda i: datetime(year=i[0],month=i[1],day=1))

## Cache Distribution
timestamp_distribution_df.to_csv(f"{RESULTS_DIR}timestamp_distribution_monthly.csv", index=False)

## Pivot
timestamp_pivot = pd.pivot_table(timestamp_distribution_df,
                                 index=["month"],
                                 columns=["dataset",CONDITION],
                                 values=["count"],
                                 aggfunc=max)["count"].fillna(0).T

## Create Heat Map
fig, ax = plt.subplots()
plot_data = timestamp_pivot.apply(lambda row: row / max(row), axis=1).copy()
dates = plot_data.columns.tolist()
xticks = [i+.5 for i, d in enumerate(dates) if d.month==1]
xticklabels = [d.year for d in dates if d.month==1]
yticklabels = ["{} ({})".format(i[0],i[1]) for i in plot_data.index.tolist()]
yticks = list(range(len(plot_data)))
m = ax.imshow(plot_data,
          cmap=plt.cm.Purples,
          aspect="auto",
          interpolation="nearest")
cbar = fig.colorbar(m)
cbar.set_label("% of Row Maximum")
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=30, ha="center")
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylim(len(plot_data)-.5,-.5)
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}time_distributions_heatmap.png")
plt.close()

## Create Multiple Line Plot
fig, ax = plt.subplots(2, 3, figsize=(10,5.8), sharex=True)
ax = flatten(ax)
for d, dataset in enumerate(DATA_SETS):
    for l, label in enumerate(["control",CONDITION]):
        plot_data = timestamp_distribution_df.loc[(timestamp_distribution_df["dataset"]==dataset)&
                                                  (timestamp_distribution_df[CONDITION]==label)]
        plot_data = plot_data.sort_values("month", ascending=True).copy().reset_index(drop=True)
        ax[d].plot(plot_data["month"],
                   plot_data["count"],
                   label = label,
                   color=f"C{l}")
        ax[d].set_xlabel("Date")
        ax[d].set_title(dataset, loc = "left")
        if d == 0 or d == 3:
            ax[d].set_ylabel("# Posts")
        ax[d].set_xlim(timestamp_distribution_df["month"].min()-timedelta(60),
                       timestamp_distribution_df["month"].max()+timedelta(60))
        ax[d].set_yscale("symlog")
ax[-1].legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.autofmt_xdate()
fig.subplots_adjust(wspace=.3, bottom=.15)
plt.savefig(f"{PLOT_DIR}time_distributions.png")
plt.close()

## User Time Boundaries
user_time_boundaries_df = pd.DataFrame.from_dict(user_time_boundaries,
                                                 orient="index",
                                                 columns=["min_date","max_date","num_posts"])
user_time_boundaries_df["n_days"] = user_time_boundaries_df[["min_date","max_date"]].diff(axis=1).apply(lambda i: i["max_date"].days, axis=1)
user_time_boundaries_df = pd.merge(user_time_boundaries_df,
                                   datasets_df.set_index("source")[["dataset",CONDITION]],
                                   left_index=True,
                                   right_index=True)

## Cache Boundaries
user_time_boundaries_df.reset_index().rename(columns={"index":"source"}).to_csv(
                                 f"{RESULTS_DIR}user_time_boundaries.csv", index=False)

## Plot Time Boundary Statistics
fig, ax = plt.subplots(2,3,figsize=(10,5.8))
ax = flatten(ax)
for d, dataset in enumerate(DATA_SETS):
    for l, label in enumerate(["control",CONDITION]):
        plot_data = user_time_boundaries_df.loc[(user_time_boundaries_df["dataset"]==dataset)&
                                                (user_time_boundaries_df[CONDITION]==label)]
        ax[d].hist(plot_data["n_days"],
                   bins=30,
                   color=f"C{l}",
                   alpha=.4,
                   label=label,
                   density=False,
                   weights=np.ones(len(plot_data)) / len(plot_data))
    ax[d].set_xlabel("# Days of Data")
    ax[d].set_title(dataset, loc = "left")
    if d == 0 or d == 3:
        ax[d].set_ylabel("Proportion of Users")
ax[-1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.subplots_adjust(wspace=.3)
plt.savefig(f"{PLOT_DIR}days_per_user_distribution.png")
plt.close()

## Compute Frequency Statistics
user_time_boundaries_df["posts_per_day"] = user_time_boundaries_df.apply(lambda row: row["num_posts"]/max(row["n_days"],1),axis=1)

## Plot Post Frequency Statistics
fig, ax = plt.subplots(2,3,figsize=(10,5.8))
ax = flatten(ax)
for d, dataset in enumerate(DATA_SETS):
    for l, label in enumerate(["control",CONDITION]):
        plot_data = user_time_boundaries_df.loc[(user_time_boundaries_df["dataset"]==dataset)&
                                                (user_time_boundaries_df[CONDITION]==label)]
        ax[d].hist(np.log10(plot_data["posts_per_day"]),
                   bins=30,
                   color=f"C{l}",
                   alpha=.4,
                   label=label,
                   density=False,
                   weights=np.ones(len(plot_data)) / len(plot_data))
    ax[d].set_xlabel("Posts Per Day (log10)")
    ax[d].set_title(dataset, loc = "left")
    if d == 0 or d == 3:
        ax[d].set_ylabel("Proportion of Users")
ax[-1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.subplots_adjust(wspace=.3)
plt.savefig(f"{PLOT_DIR}posts_per_day_distribution.png")
plt.close()

logger.info("Script Complete")