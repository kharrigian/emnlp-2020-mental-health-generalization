
"""
Compare sumamry statistics between data sets (e.g. posts per user)
and evaluate how unique each data set vocabulary is.
"""

## Directories
PLOT_DIR = "./plots/vocabulary_similarity/"
DATA_DIR = "./data/processed/"
CACHE_DIR = "./data/results/cache/vocabulary_similarity/"

## Processing
VOCAB_CHUNKSIZE = 30
NUM_JOBS = 8
RERUN = False

## Data Sets and Balancing/Downsampling Parameters
DATA_SETS = {
        "clpsych_deduped":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":False,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2011-01-01","max_date":"2013-12-01"}
        },
        "multitask":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":False,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2013-01-01","max_date":"2016-01-01"}
        },
        "wolohan":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2014-01-01","max_date":"2020-01-01"}
        },
        "rsdd":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2008-01-01","max_date":"2017-01-01"}
        },
        "smhd":{
                        "downsample":False,
                        "downsample_size":100,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2010-01-01","max_date":"2018-01-01"}
        },
}

## Meta
RANDOM_SEED = 42

########################
### Imports
########################

## Standard Libraries
import os
import sys
from collections import Counter

## External Libraries
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction import DictVectorizer

## Local
from mhlib.util.helpers import flatten
from mhlib.model import train
from mhlib.model.vocab import Vocabulary
from mhlib.util.logging import initialize_logger

########################
### Globals
########################

## Initialize Logger
LOGGER = initialize_logger()

## Plot Directory
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

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
    col_subset = ["user_id_str","depression","gender","age","split","num_posts","num_words","dataset","source"]
    df = df[[c for c in col_subset if c in df.columns]].copy()
    return df

def plot_distributions(df, value, hist=False):
    """
    Plot Distribution of Values Across Data Sets

    Args:
        df (pandas DataFrame): Metadata DataFrame
        value (str): Numeric value column in the dataframe
        hist (bool): If True, plot a histogram. Otherwise, plot a boxplot.
    
    Returns:
        fig, ax (matplotlib objects): Distribution Visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=False)
    axes = flatten(axes)
    for ds, ax in zip(DATA_SETS, axes):
        for l, lbl in enumerate(["control","depression"]):
            ds_df = df.loc[(df["dataset"]==ds)&(df["depression"]==lbl)]
            if hist:
                ax.hist(np.log10(ds_df[value] + 1e-10),
                        bins = 30,
                        color = f"C{l}",
                        alpha = .5,
                        density = True,
                        label = lbl)
            else:
                counts = ds_df[value].value_counts()
                ax.scatter(counts.index,
                           counts.values,
                           color = f"C{l}",
                           alpha = .5,
                           lbl = lbl)
        ax.set_title(ds, loc = "left", fontsize = 10)
        if not hist:
            ax.set_yscale("log")
            ax.set_xscale("log")
        ax.set_xlim(left=0)
    axes[2].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True, fontsize=6)
    axes[0].set_ylabel("% of Group", fontsize=10)
    axes[3].set_ylabel("% of Group", fontsize=10)
    for ax in axes[3:]:
        if hist:
            ax.set_xlabel(f"{value} per ind. (log10)")
        else:
            ax.set_xlabel(f"{value} per ind.")
    fig.tight_layout()
    return fig, axes        

def jaccard_index(l1, l2):
    """
    Compute the jaccard index between two arrays.
    J(X,Y) = |X∩Y| / |X∪Y|

    Args:
        l1 (iterable): First set of items
        l2 (iterable): Second set of items
    
    Returns:
        J_index (float): Jaccard score between sets
    """
    l1 = set(l1)
    l2 = set(l2)
    J_index = len(l1 & l2) / len(l1 | l2)
    return J_index

def plot_heatmap(array,
                 xticks=[],
                 yticks=[],
                 title="Jaccard Similarity"):
    """
    Plot a Heat Map

    Args:
        array (2d-array): Numpy array
        xticks (list): X-tick labels
        yticks (list): Y-tick labels
        title (str): Title of the Plot
    """
    mask = np.ones_like(array, dtype=float)
    mask[np.triu_indices_from(mask)] = np.nan
    mask[np.diag_indices_from(mask)] = 1
    fig, ax = plt.subplots(figsize=(10,5.6))
    mat = ax.imshow(array * mask,
                    cmap = plt.cm.Blues,
                    alpha = .5,
                    aspect = "auto",
                    interpolation = "nearest")
    ax.set_xticks(list(range(array.shape[1])))
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    ax.set_yticks(list(range(array.shape[0])))
    ax.set_yticklabels(yticks)
    ax.set_title(title, fontsize = 10)
    for i, ival in enumerate(array * mask):
        for j, jval in enumerate(ival):
            if pd.isnull(jval):
                continue
            ax.text(j, i, "{:.2f}".format(jval), fontsize=8, va="center", ha="center")
    ax.set_ylim(len(yticks)-.5, -.5)
    fig.tight_layout()
    return fig, ax

def create_dict_vectorizer(vocab_terms):
    """
    Create a Dict Vectorizer object given a fixed vocabulary

    Args:
        vocab_terms (list): Sorted set of vocabulary terms
    
    Returns:
        _count2vec (DictVectorizer): Sklearn vectorizer
    """
    ngram_to_idx = dict((n, i) for i, n in enumerate(sorted(vocab_terms)))
    _count2vec = DictVectorizer(separator=":")
    _count2vec.vocabulary_ = ngram_to_idx.copy()
    rev_dict = dict((y, x) for x, y in ngram_to_idx.items())
    _count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
    return _count2vec

def overlap_index(l1, l2):
    """
    Returns the percentage of l1 that is in l2
    """
    val = len(set(l1) & set(l2)) / len(set(l1))
    return val

########################
### Load Data
########################

LOGGER.info("Loading Dataset Metadata")

## Load Datasets Jointly
datasets_df = []
for ds in DATA_SETS:
    ## Load Metadata
    ds_df = train.load_dataset_metadata(ds,
                                        "depression",
                                        RANDOM_SEED)
    ## Balance/Downsample
    if DATA_SETS[ds]["rebalance"]:
        ds_df = train._rebalance(ds_df,
                                 "depression",
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
### Analysis of Content Per User
########################

LOGGER.info("Plotting Volume Statistics")

## Add Statistics
datasets_df["words_per_post"] = datasets_df["num_words"] / datasets_df["num_posts"]

## Distributions Per Individual
for stat in ["num_posts", "num_words", "words_per_post"]:
    ## Distributions
    fig, ax = plot_distributions(datasets_df,
                                 value = stat,
                                 hist = True)
    fig.savefig(f"{PLOT_DIR}{stat}_distribution.png")
    plt.close()
    ## Summary Stats
    fig = sns.boxplot(x = "dataset",
                      y = stat,
                      hue = "depression",
                      data = datasets_df,
                      showfliers = False)
    plt.ylabel(f"{stat} per Ind.")
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}{stat}_boxplot.png")
    plt.close()

########################
### Learn Vocabularies
########################

LOGGER.info("Learning Vocabularies")

## Loading Arguments
vocab_kwargs = {
    "filter_negate":True,
    "filter_upper":True,
    "filter_punctuation":True,
    "filter_numeric":True,
    "filter_user_mentions":True,
    "filter_url":True,
    "filter_retweet":True,
    "filter_stopwords":False,
    "keep_pronouns":True,
    "preserve_case":False,
    "emoji_handling":None,
    "max_vocab_size":None,
    "min_token_freq":10,
    "max_token_freq":None,
    "ngrams":(1,1),
    "max_tokens_per_document":None,
    "max_documents_per_user":None,
    "binarize_counter":True,
    "filter_mh_subreddits":"all",
    "filter_mh_terms":"smhd",
}

## Vocabulary Caches
vocabs = dict((ds,{}) for ds in DATA_SETS)
vocabs_merged = {}
sample_sizes = dict((ds,{}) for ds in DATA_SETS)
sample_sizes_merged = {}

if not os.path.exists(f"{CACHE_DIR}vocabs.joblib") or not os.path.exists(f"{CACHE_DIR}sample_sizes.joblib") or RERUN:
    ## Identify Vocabulary in Each Dataset
    for dataset in DATA_SETS:
        for condition in ["control","depression"]:
            LOGGER.info(f"Learning Vocabulary for `{dataset} - {condition}`")
            ## Isolate Files for Dataset + Condition
            dataset_files = datasets_df.loc[(datasets_df["dataset"]==dataset)&
                                            (datasets_df["depression"]==condition)]["source"].tolist()
            ## Cache Sample Size
            sample_sizes[dataset][condition] = len(dataset_files)
            ## Learn Vocabulary
            dataset_vocab = Vocabulary(**vocab_kwargs)
            dataset_vocab.fit(dataset_files,
                            chunksize=VOCAB_CHUNKSIZE,
                            jobs=NUM_JOBS,
                            min_date=DATA_SETS[dataset]["date_boundaries"]["min_date"],
                            max_date=DATA_SETS[dataset]["date_boundaries"]["max_date"])
            ## Cache Vocabulary
            vocabs[dataset][condition] = Counter(dict((x, y) for x, y in dataset_vocab._vocab_count.items() if y >= vocab_kwargs["min_token_freq"]))
    ## Cache Reults
    _ = joblib.dump(vocabs, f"{CACHE_DIR}vocabs.joblib")
    _ = joblib.dump(sample_sizes, f"{CACHE_DIR}sample_sizes.joblib")
    _ = joblib.dump(DATA_SETS, f"{CACHE_DIR}data_sets.joblib")
else:
    ## Load Results
    vocabs = joblib.load(f"{CACHE_DIR}vocabs.joblib")
    sample_sizes = joblib.load(f"{CACHE_DIR}sample_sizes.joblib")
    DATA_SETS = joblib.load(f"{CACHE_DIR}data_sets.joblib")

## Merge Conditions Within Each Data Set
for ds, ds_vocab in vocabs.items():
    vocabs_merged[ds] =  ds_vocab["control"] + ds_vocab["depression"]
    sample_sizes_merged[ds] = sample_sizes[ds]["control"] + sample_sizes[ds]["depression"]

########################
### Overlap
########################

LOGGER.info("Computing Overlap Scores (Per Data Set)")

## Dataset, Conditions
datasets_and_conds = flatten([[(ds, cond) for cond in ["control","depression"]] for ds in DATA_SETS])

## Get Vocabulary Sizes (And Overlap)
overlap_sizes =  np.zeros((len(DATA_SETS), len(DATA_SETS)))
for d1, dataset1 in enumerate(DATA_SETS):
    for d2, dataset2 in enumerate(DATA_SETS):
        overlap_sizes[d1, d2] = len(set(vocabs_merged[dataset1].keys()) & set(vocabs_merged[dataset2].keys()))
overlap_sizes = pd.DataFrame(overlap_sizes,
                              index=DATA_SETS,
                              columns=DATA_SETS)

LOGGER.info("~"*50 + "\nOverlap Sizes\n" + "~"*50)
LOGGER.info(overlap_sizes.astype(int))
                        
## Compute Overlap (Per Dataset)
overlap_scores = np.zeros((len(DATA_SETS), len(DATA_SETS)))
for d1, dataset1 in enumerate(DATA_SETS):
    for d2, dataset2 in enumerate(DATA_SETS):
        overlap_scores[d1, d2] = overlap_index(l1=set(vocabs_merged[dataset1].keys()),
                                               l2=set(vocabs_merged[dataset2].keys()))
overlap_scores = pd.DataFrame(overlap_scores,
                              index=DATA_SETS,
                              columns=DATA_SETS)

LOGGER.info("~"*50 + "\nOverlap Scores\n" + "~"*50)
LOGGER.info(overlap_scores)
                        
## Get Vocabulary Sizes (Dataset and Condition)
overlap_sizes_by_condition = np.zeros((len(DATA_SETS)*2, len(DATA_SETS)*2))
for d1, (dataset1, cond1) in enumerate(datasets_and_conds):
    for d2, (dataset2, cond2) in enumerate(datasets_and_conds):
        overlap_sizes_by_condition[d1, d2] = len(set(vocabs[dataset1][cond1]) & set(vocabs[dataset2][cond2]))
overlap_sizes_by_condition = pd.DataFrame(overlap_sizes_by_condition,
                              index=datasets_and_conds,
                              columns=datasets_and_conds)

## Compute Overlap (Per Dataset and Condition)
overlap_scores_by_condition = np.zeros((len(DATA_SETS)*2, len(DATA_SETS)*2))
for d1, (dataset1, cond1) in enumerate(datasets_and_conds):
    for d2, (dataset2, cond2) in enumerate(datasets_and_conds):
        overlap_scores_by_condition[d1, d2] = \
                                    overlap_index(l1=set(vocabs[dataset1][cond1]),
                                                  l2=set(vocabs[dataset2][cond2]))
overlap_scores_by_condition = pd.DataFrame(overlap_scores_by_condition,
                                           index=datasets_and_conds,
                                           columns=datasets_and_conds)

########################
### Jaccard Similarity
########################

LOGGER.info("Computing Jaccard Scores (Per Data Set)")

## Jaccard Index (Across Datasets)
jaccard_scores = np.zeros((len(DATA_SETS), len(DATA_SETS)))
for d1, dataset1 in enumerate(DATA_SETS):
    for d2, dataset2 in enumerate(DATA_SETS):
        jaccard_scores[d1, d2] = jaccard_index(l1=set(vocabs_merged[dataset1].keys()),
                                               l2=set(vocabs_merged[dataset2].keys()))

LOGGER.info("Computing Jaccard Scores (Per Data Set and Condition)")

## Jaccard Index (Across Datasets and Between conditions)
jaccard_scores_by_condition = np.zeros((len(DATA_SETS)*2, len(DATA_SETS)*2))
for d1, (dataset1, cond1) in enumerate(datasets_and_conds):
    for d2, (dataset2, cond2) in enumerate(datasets_and_conds):
        jaccard_scores_by_condition[d1, d2] = \
                                    jaccard_index(l1=set(vocabs[dataset1][cond1]),
                                                  l2=set(vocabs[dataset2][cond2]))

LOGGER.info("Computing Jaccard Scores (~ Threshold)")

## Jaccard Index ~ Data Set Size (Merged)
thresholds = np.arange(10, 21)
sim_by_thres = []
for d1, dataset1 in enumerate(DATA_SETS):
    v1 = pd.Series(vocabs_merged[dataset1])
    for d2, dataset2 in enumerate(DATA_SETS):
        if d2 >= d1:
            continue
        v2 =  pd.Series(vocabs_merged[dataset2])
        for thres in thresholds:
            sim_by_thres.append([dataset1, 
                                 dataset2,
                                 thres,
                                 jaccard_index(set(v1[v1 > thres].index), set(v2[v2>thres].index))])
sim_by_thres = pd.DataFrame(sim_by_thres, columns = ["d1","d2","thresh","jaccard"])

########################
### Ratio Analysis
########################

## Get Vocab Proportions
vocab_ratios_df = []
for dataset in DATA_SETS:
    dataset_vocab_df = []
    for condition in ["control","depression"]:
        dcond = pd.Series(vocabs[dataset][condition]).to_frame(condition)
        dcond = dcond / sample_sizes[dataset][condition]
        dataset_vocab_df.append(dcond)
    dataset_vocab_df = pd.concat(dataset_vocab_df, axis=1).dropna()
    dataset_vocab_df[f"{dataset}_ratio"] = np.log(dataset_vocab_df["depression"] / dataset_vocab_df["control"])
    vocab_ratios_df.append(dataset_vocab_df[[f"{dataset}_ratio"]])
vocab_ratios_df = pd.concat(vocab_ratios_df, axis=1)

## Compute Correlation Statistics
vocab_ratios_corr = vocab_ratios_df.corr()

## Pair Plot
f = sns.pairplot(vocab_ratios_df, kind="reg", plot_kws={"scatter_kws":{"alpha":0.5, "s":5}})
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}vocabulary_group_ratio_correlation_pairplot.png", dpi=300)
plt.close()

## Correlation Summary
fig, ax = plot_heatmap(vocab_ratios_corr.values,
                       xticks=DATA_SETS,
                       yticks=DATA_SETS,
                       title="Ratio Correlation")
plt.savefig(f"{PLOT_DIR}vocabulary_group_ratio_correlation.png", dpi=300)
plt.close()

########################
### Plot Vocabulary Statistics
########################

LOGGER.info("Plotting Vocabulary Statistics")

## Vocab Size (Relative to Threshold)
fig, ax = plt.subplots(figsize=(10,5.6))
for d, dataset in enumerate(DATA_SETS):
    series_count = pd.Series(vocabs_merged[dataset]).value_counts().sort_index()
    remaining_by_thresh = series_count[::-1].cumsum()[::-1]
    remaining_by_thresh = remaining_by_thresh.reindex(np.arange(1, remaining_by_thresh.index.max()+1)).fillna(method="bfill")
    ax.plot(remaining_by_thresh.index,
            remaining_by_thresh.values,
            color = f"C{d}",
            alpha = .5,
            label = dataset)
ax.set_xlabel("Min Occurrence (Threshold)")
ax.set_ylabel("Vocabulary Size")
ax.legend(bbox_to_anchor=(1.1, 1.05), loc="upper right", fancybox=True, shadow=True, fontsize=12)
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}vocabulary_size_by_threshold.png")
plt.close()

## Plot Jaccard Index (Across Datasets)
fig, ax = plot_heatmap(jaccard_scores,
                       xticks=DATA_SETS,
                       yticks=DATA_SETS,
                       title="Jaccard Similarity")
plt.savefig(f"{PLOT_DIR}vocabulary_jaccard_similarity.png")
plt.close()

## Plot Jaccard Index (Across Datasets and Between Conditions)
fig, ax = plot_heatmap(jaccard_scores_by_condition,
                       xticks=datasets_and_conds,
                       yticks=datasets_and_conds,
                       title="Jaccard Similarity")
plt.savefig(f"{PLOT_DIR}vocabulary_by_condition_jaccard_similarity.png")
plt.close()

## Plot Jaccard Index vs. Threshold
for ds in DATA_SETS:
    ds_data = sim_by_thres.loc[(sim_by_thres["d1"]==ds)|(sim_by_thres["d2"]==ds)]
    fig, ax = plt.subplots(figsize=(10,5.8))
    for d2, ds2 in enumerate(DATA_SETS):
        if ds2 == ds:
            continue
        ds2_data = ds_data.loc[(ds_data["d1"]==ds2)|(ds_data["d2"]==ds2)]
        ax.plot(ds2_data["thresh"],
                ds2_data["jaccard"],
                color=f"C{d2}",
                label=ds2)
    ax.legend(loc="lower right")
    ax.set_title(f"Comparison Data Set: {ds}", loc = "left")
    ax.set_xlabel("Vocabulary Occurrence Threshold")
    ax.set_ylabel("Jaccard Index")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}jaccard_threshold_{ds}.png")
    plt.close()