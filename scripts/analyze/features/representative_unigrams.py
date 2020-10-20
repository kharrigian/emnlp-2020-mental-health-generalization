
## Directories
PLOT_DIR = "./plots/representative_features/"
DATA_DIR = "./data/processed/"
CACHE_DIR = "./data/results/cache/representative_features/"

## Condition
CONDITION = "depression"

## Processing
VOCAB_CHUNKSIZE = 30
NUM_JOBS = 8

## Data Sets and Balancing/Downsampling Parameters
DATA_SETS = {
        "clpsych_deduped":{
                        "downsample":True,
                        "downsample_size":580,
                        "rebalance":False,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2011-01-01","max_date":"2013-12-01"}
        },
        "multitask":{
                        "downsample":True,
                        "downsample_size":580,
                        "rebalance":False,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2013-01-01","max_date":"2016-01-01"}
        },
        "wolohan":{
                        "downsample":True,
                        "downsample_size":580,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2014-01-01","max_date":"2020-01-01"}
        },
        "rsdd":{
                        "downsample":True,
                        "downsample_size":580,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2008-01-01","max_date":"2017-01-01"}
        },
        "smhd":{
                        "downsample":True,
                        "downsample_size":580,
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

## Standard Library
import os
from copy import deepcopy

## External Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

## Local
from mhlib.model import train
from mhlib.util.logging import initialize_logger
from mhlib.util.helpers import flatten
from mhlib.model.file_vectorizer import File2Vec
from mhlib.model.feature_extractors import LIWCTransformer
from mhlib.model.feature_selectors import KLDivergenceSelector

########################
### Globals
########################

## Initialize Logger
LOGGER = initialize_logger()

## Plot Directory
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

## Cache Dir
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
    col_subset = ["user_id_str",CONDITION,"gender","age","split","num_posts","num_words","dataset","source"]
    df = df[[c for c in col_subset if c in df.columns]].copy()
    return df

########################
### Load Data
########################

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

## Label Mappings
label_maps = {}
for dataset in DATA_SETS:
    dataset_subset = datasets_df.loc[(datasets_df["dataset"]==dataset)]
    label_maps[dataset] = dataset_subset.set_index("source")[CONDITION].to_dict()

########################
### Vectorization
########################

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
    "filter_mh_terms":"smhd"
}

LOGGER.info("Learning Full Vocabulary")

## Learn Vocabulary
filenames = flatten(label_maps.values())
f2v = File2Vec(vocab_kwargs=vocab_kwargs)
f2v.vocab = f2v.vocab.fit(filenames, chunksize=VOCAB_CHUNKSIZE, jobs=NUM_JOBS)

LOGGER.info("Vectorizing Data")

## Vectorize Files
f2v._initialize_dict_vectorizer()
filenames, X = f2v._vectorize_files(filenames, 8)

## Create Target Classes
dataset_filemap = datasets_df.set_index("source")["dataset"].to_dict()
label_filemap = datasets_df.set_index("source")[CONDITION].to_dict()
y_dataset = list(map(lambda f: dataset_filemap[f], filenames))
y_label = list(map(lambda f: label_filemap[f], filenames))

## Filtering Function
def create_subset(X,
                  y_dataset,
                  y_label,
                  data_classes=list(DATA_SETS.keys()),
                  dep_classes=[CONDITION],
                  split_ds=True):
    """

    """
    ## Subset
    dep_ind = [i for i, c in enumerate(y_label) if c in dep_classes]
    data_ind = [i for i, c in enumerate(y_dataset) if c in data_classes]
    subset_ind = sorted(set(dep_ind)&set(data_ind))
    ## Y Formulation
    y_dep_subset = [y_label[i] for i in subset_ind]
    if split_ds:
        y_ds_subset = [y_dataset[i] for i in subset_ind]
    else:
        y_ds_subset = ["ds"] * len(subset_ind)
    y = list(zip(y_ds_subset, y_dep_subset))
    classes = sorted(set(y))
    y_map = dict((y, x) for x, y in enumerate(classes))
    y = np.array(list(map(lambda i: y_map[i], y)))
    ## Subset
    X_subset = X[subset_ind].copy()
    return X_subset, y, classes

########################
### Experimment 1: Rank of Features
########################

LOGGER.info("Computing Feature Ranks")

## Cache for Feature Scores
all_condition_scores = []
all_control_scores = []
## Cycle Through Data Sets
for dataset in DATA_SETS.keys():
    LOGGER.info(f"\nGetting Feature Rank For {dataset}")
    ## Select Subset
    X_subset, y_subset, classes = create_subset(X,
                                                y_dataset,
                                                y_label,
                                                data_classes=[dataset],
                                                dep_classes=[CONDITION,"control"],
                                                split_ds=False)
    ## Initialize Feature Selector
    selector = KLDivergenceSelector(deepcopy(f2v.vocab),
                                    min_support = 30,
                                    add_lambda = 0.0001,
                                    beta = 0.2,
                                    symmetric=False)
    ## Fit 
    selector = selector.fit(X_subset, y_subset)
    ## Compile Scores
    scores = pd.DataFrame(selector._div_score,
                        columns=selector._vocabulary_terms,
                        index=classes).T
    condition_scores = scores[[("ds",CONDITION)]].copy()
    control_scores = scores[[("ds","control")]].copy()
    condition_scores["freq"] = selector._term_freq
    control_scores["freq"] = selector._term_freq
    condition_scores["dataset"] = dataset
    control_scores["dataset"] = dataset
    condition_scores = condition_scores.reset_index().rename(columns={"index":"feature",
                                                                       ("ds",CONDITION):"score"})
    control_scores = control_scores.reset_index().rename(columns={"index":"feature",
                                                                 ("ds","control"):"score"})
    ## Append
    all_condition_scores.append(condition_scores)
    all_control_scores.append(control_scores)

## Concatenate Scores
all_condition_scores = pd.concat(all_condition_scores).reset_index(drop=True)
all_control_scores = pd.concat(all_control_scores).reset_index(drop=True)

## Pivot Table
all_condition_scores_pivot = pd.pivot_table(all_condition_scores,
                                  index="feature",
                                  columns="dataset",
                                  values=["score","freq"],
                                  aggfunc=max)
all_control_scores_pivot = pd.pivot_table(all_control_scores,
                                  index="feature",
                                  columns="dataset",
                                  values=["score","freq"],
                                  aggfunc=max)

## Feature Overlap
condition_scores_ranked = all_condition_scores_pivot["score"].rank(ascending=False)
control_scores_ranked = all_control_scores_pivot["score"].rank(ascending=False)
top_overlap = []
for top_k in range(100, X.shape[1], 100):
    dep_overlap = condition_scores_ranked.loc[(condition_scores_ranked <= top_k).all(axis=1)].index.tolist()
    con_overlap = control_scores_ranked.loc[(control_scores_ranked <= top_k).all(axis=1)].index.tolist()
    top_overlap.append((top_k, len(dep_overlap), len(con_overlap), dep_overlap, con_overlap))
top_overlap = pd.DataFrame(top_overlap,columns=["top_k",CONDITION,"control","dep_ex","con_ex"])

## Plot Feature Overlap
fig, ax = plt.subplots(figsize=(10,5.8))
top_overlap.set_index("top_k").plot(ax=ax)
ax.set_xlabel("Top-k Threshold")
ax.set_ylabel("Overlap Over Data Sets")
plt.yscale("symlog")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}top_feature_overlap_over_thresh.png")
plt.close()

## Rank Table
n_reps = 500
min_usage = 10
dep_representative_vals = {}
con_representative_vals = {}
for dataset in DATA_SETS.keys():
    dep_reps = condition_scores_ranked[dataset].loc[all_condition_scores_pivot["freq"][dataset]>=min_usage].nsmallest(n_reps).index.tolist()
    con_reps = control_scores_ranked[dataset].loc[all_control_scores_pivot["freq"][dataset]>=min_usage].nsmallest(n_reps).index.tolist()
    dep_representative_vals[dataset] = dep_reps
    con_representative_vals[dataset] = con_reps
dep_representative_vals = pd.DataFrame(dep_representative_vals)
con_representative_vals = pd.DataFrame(con_representative_vals)

## Dump Representative Vals
dep_representative_vals.to_csv(f"{CACHE_DIR}representative_{CONDITION}_features.csv")
con_representative_vals.to_csv(f"{CACHE_DIR}representative_control_features.csv")