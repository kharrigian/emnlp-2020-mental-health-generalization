

"""
Compare distributions of LIWC and GloVe representations
"""

###########################
### Configuration
###########################

## Directories
DATA_DIR = "./data/processed/"
RESULTS_DIR = "./data/results/cache/external_resources/"
PLOT_DIR = "./plots/external_resources/"

## Flag to Re-run even if data exists
RERUN = True

## Data Sets and Balancing/Downsampling Parameters
DATA_SETS = {
        "clpsych_deduped":{
                        "downsample":True,
                        "downsample_size":500,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2011-01-01","max_date":"2013-12-01"}
        },
        "multitask":{
                        "downsample":True,
                        "downsample_size":500,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2013-01-01","max_date":"2016-01-01"}
        },
        "wolohan":{
                        "downsample":True,
                        "downsample_size":500,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2014-01-01","max_date":"2020-01-01"}
        },
        "rsdd":{
                        "downsample":True,
                        "downsample_size":500,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2008-01-01","max_date":"2017-01-01"}
        },
        "smhd":{
                        "downsample":True,
                        "downsample_size":500,
                        "rebalance":True,
                        "class_ratio":[1,1],
                        "date_boundaries":{"min_date":"2010-01-01","max_date":"2018-01-01"}
        },
}

## Vocabulary Parameters
VOCAB_PARAMS = {
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
    "emoji_handling":"strip",
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

## Meta
RANDOM_SEED = 42
NUM_PROCESSES = 8

## Default Data Set Order
DS_ORDER = ["clpsych_deduped","multitask","rsdd","smhd","wolohan"]

## Data Set Names
DS_NAMES = {
    "clpsych_deduped":"CLPsych",
    "clpsych":"ClPysch",
    "multitask":"Multi-\nTask",
    "rsdd":"RSDD",
    "smhd":"SMHD",
    "wolohan":"Topic-\nRestricted"
}

###########################
### Imports
###########################

## Standard Libraries
import os
import sys
import re
import gzip
import json
from copy import deepcopy
from multiprocessing import Pool

## External Libraries
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise
from umap import UMAP

## Local
from mhlib.model import train
from mhlib.model.vocab import Vocabulary
from mhlib.model.file_vectorizer import File2Vec
from mhlib.model.feature_extractors import (LIWCTransformer,
                                            EmbeddingTransformer)
from mhlib.util.helpers import flatten
from mhlib.util.logging import initialize_logger

###########################
### Globals
###########################

## Logger
LOGGER = initialize_logger()

## Directories
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

###########################
### Functions
###########################

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

## Visualize Hit Rates
def visualize_hit_rates(hit_rate,
                        y_dataset,
                        y_depression):
    """

    """
    ## Reshape Hit Rate
    hit_rate = np.array(hit_rate).T[0]
    ## Create Plot
    fig, ax = plt.subplots(2, 3, figsize=(10,5.8))
    ax = flatten(ax)
    min_y, max_y = 1, -1
    for d, dataset in enumerate(sorted(set(y_dataset))):
        for t, target in enumerate(["Control","Depression"]):
            ## Identify Subset
            mask = np.nonzero(np.logical_and(y_dataset==dataset, y_depression==t))[0]
            ## Compute Stats
            mean = np.nanmean(hit_rate[mask])
            std = np.nanstd(hit_rate[mask])
            support = (~np.isnan(hit_rate[mask])).sum()
            standard_error = std / np.sqrt(support - 1)
            if mean - standard_error < min_y:
                min_y = mean - standard_error
            if mean + standard_error > max_y:
                max_y = mean + standard_error
            ## Plot Histogram
            ax[d].hist(hit_rate[mask],
                       bins=20,
                       color=f"C{t}",
                       density=True,
                       alpha=0.5,
                       label=target)
            ## Plot Summary Stats
            ax[-1].bar(d + 0.025 + t*0.475,
                       mean,
                       yerr=standard_error,
                       label=target if d == 0 else "",
                       color=f"C{t}",
                       width=0.475,
                       align="edge",
                       alpha=0.5)
        ax[d].legend(loc="upper right", frameon=True, fontsize=6)
        ax[d].set_title(dataset.replace("_"," ").upper(), loc="left", fontsize=8, fontweight="bold")
        ax[d].set_ylabel("Density", fontsize=6, fontweight="bold")
        ax[d].set_xlabel("Hit Rate", fontsize=6, fontweight="bold")
    ax[-1].set_xticks(np.arange(5)+0.5)
    ax[-1].set_xticklabels([i.replace("_","\n").upper() for i in sorted(set(y_dataset))],
                           fontsize=6)
    ax[-1].legend(loc="upper right", frameon=True, fontsize=6)
    ax[-1].set_ylim(min_y * .975, max_y * 1.025)
    ax[-1].set_ylabel("Hit Rate", fontsize=6, fontweight="bold")
    fig.tight_layout()
    return fig, ax

###########################
### Load Dataset Metadata
###########################

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

## Unique Data Sets
unique_data_sets = list(DATA_SETS.keys())

###########################
### Bag of Words Representation
###########################

## File To Save
external_resource_cache_file = f"{RESULTS_DIR}external_resource_base.joblib"

## Look for/Load Existing Data
if os.path.exists(external_resource_cache_file) and not RERUN:
    ## Load File
    ext_data = joblib.load(external_resource_cache_file)
    ## Parse Elements
    X = ext_data["X"]
    y_depression = ext_data["y_depression"]
    y_dataset = ext_data["y_dataset"]
    f2v = ext_data["f2v"]
else:
    ## Cache of Objects Per Data Set
    X = []
    y_depression = []
    y_dataset = []
    ## Initialize Vocabulary Object
    f2v = File2Vec(vocab_kwargs=VOCAB_PARAMS,
                   favor_dense=False)
    ## Learn Vocabulary
    f2v.vocab = f2v.vocab.fit(datasets_df["source"].tolist(),
                            chunksize=30,
                            jobs=NUM_PROCESSES)
    ## Initialize Vectorizer
    f2v._initialize_dict_vectorizer()
    ## Cycle Through Data Sets
    for dataset in DATA_SETS.keys():
        ## Get Bag of Words Representation
        LOGGER.info(f"Vectorizing Data from {dataset}")
        filenames, _X= f2v._vectorize_files(datasets_df.loc[datasets_df["dataset"]==dataset]["source"].tolist(),
                                            jobs=NUM_PROCESSES,
                                            min_date=DATA_SETS[dataset]["date_boundaries"]["min_date"],
                                            max_date=DATA_SETS[dataset]["date_boundaries"]["max_date"])
        ## Align With Depression Class
        _y_depression = (datasets_df.set_index("source").loc[filenames]["depression"]=="depression").astype(int).values
        ## Cache Objects
        X.append(_X)
        y_depression.extend(_y_depression)
        y_dataset.extend([dataset for _ in range(len(filenames))])
    ## Stack
    X = vstack(X)
    y_depression = np.array(y_depression)
    y_dataset = np.array(y_dataset)
    ## Cache
    _ = joblib.dump({"X":X,"y_depression":y_depression,"y_dataset":y_dataset,"f2v":f2v},
                    external_resource_cache_file)

###########################
### LIWC/GloVe Representations
###########################

## Initialize Transformers
liwc = LIWCTransformer(vocab=f2v.vocab,
                       norm=None)
glove = EmbeddingTransformer(vocab=f2v.vocab,
                             dim=200,
                             pooling="mean",
                             jobs=NUM_PROCESSES)

## Fit Transformers
liwc = liwc.fit(X)
glove = glove.fit(X)

## Apply Transformations
X_liwc = liwc.transform(X)
X_glove = glove.transform(X)

## Normalize LIWC by Total Word Usage
X_liwc_normed = np.divide(X_liwc,
                          np.array(X.sum(axis=1)),
                          out=np.ones_like(X_liwc) * np.nan,
                          where=np.array(X.sum(axis=1))>0)

## Normalize GloVe with L2
X_glove = normalize(X_glove, axis=1, norm="l2")

########################
### Hit Rate
########################

LOGGER.info("Computing and Visualizing Hit Rates")

## Glove Hit Rate
glove_match_vector = csr_matrix(np.any(glove.embedding_matrix!=0,axis=1).astype(int).reshape(1,-1))
glove_matches = X.multiply(glove_match_vector).sum(axis=1)
glove_hit_rate = np.divide(glove_matches,
                           X.sum(axis=1),
                           out=np.ones_like(glove_matches) * np.nan,
                           where=X.sum(axis=1)>0)

## Plot GloVe Hit Rate
fig, ax = visualize_hit_rates(glove_hit_rate,
                              y_dataset,
                              y_depression)
fig.savefig(f"{PLOT_DIR}hit_rate_GloVe.png", dpi=150)
plt.close(fig)

## LIWC Hit Rate
liwc_match_vector = csr_matrix(np.any(liwc._dim_map!=0,axis=1).astype(int))
liwc_matches = X.multiply(liwc_match_vector).sum(axis=1)
liwc_hit_rate = np.divide(liwc_matches,
                          X.sum(axis=1),
                          out=np.ones_like(liwc_matches) * np.nan,
                          where=X.sum(axis=1)>0)

## Plot LIWC Hit Rate
fig, ax = visualize_hit_rates(liwc_hit_rate,
                              y_dataset,
                              y_depression)
fig.savefig(f"{PLOT_DIR}hit_rate_LIWC.png", dpi=150)
plt.close(fig)

########################
### GloVe 2D-Projection (UMAP)
########################

LOGGER.info("Fitting UMAP Projection of GloVe Represention")

## Identify Non-null Map
nn_glove_map = np.nonzero(~(X_glove==0).all(axis=1))[0]

## Fit Representation
u = UMAP(random_state=42)
X_glove_umap = u.fit_transform(X_glove[nn_glove_map])

## Plot Projection
fig, ax = plt.subplots()
for d, dataset in enumerate(sorted(set(y_dataset))):
    for t, target in enumerate(["Control","Depression"]):
        d_mask = np.nonzero(np.logical_and(y_dataset[nn_glove_map] == dataset,
                                           y_depression[nn_glove_map]==t))[0]
        ax.scatter(X_glove_umap[d_mask, 0],
                   X_glove_umap[d_mask, 1],
                   color=f"C{d}",
                   alpha=0.25,
                   marker="o" if t == 0 else "s",
                   s=10,
                   label=dataset.replace("_"," ").upper() if t == 0 else "")
ax.legend(loc="best", frameon=True, fontsize=6)
ax.set_xlabel("Projection Axis 1", fontsize=8, fontweight="bold")
ax.set_ylabel("Projection Axis 2", fontsize=8, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}umap_GloVe.png", dpi=150)
plt.close(fig)

########################
### GloVe Sample Pairwise Similarity
########################

## GloVe Cosine Similarity
cosine_glove = pairwise.cosine_similarity(X_glove[nn_glove_map])

## Plot Similarity
fig, ax = plt.subplots(2, 3, figsize=(10,5.8))
ax = flatten(ax)
for d, dataset in enumerate(sorted(set(y_dataset))):
    ## Identify Masks
    depression_mask = np.nonzero(np.logical_and(y_dataset[nn_glove_map] == dataset,
                                                y_depression[nn_glove_map]==1))[0]
    control_mask = np.nonzero(np.logical_and(y_dataset[nn_glove_map] == dataset,
                                                y_depression[nn_glove_map]==0))[0]
    depression_different_data_mask = np.nonzero(np.logical_and(y_dataset[nn_glove_map] != dataset,
                                                               y_depression[nn_glove_map]==1))[0]
    control_different_data_mask = np.nonzero(np.logical_and(y_dataset[nn_glove_map] != dataset,
                                                            y_depression[nn_glove_map]==0))[0]
    ## Get Values
    versus_depression = cosine_glove[depression_mask][:, depression_mask].ravel()
    versus_control = cosine_glove[depression_mask][:, control_mask].ravel()
    versus_depression_external = cosine_glove[depression_mask][:, depression_different_data_mask].ravel()
    versus_control_external = cosine_glove[depression_mask][:, control_different_data_mask].ravel()
    ## Plot Boxpolt
    for i, v in enumerate([versus_depression,
              versus_control,
              versus_depression_external,
              versus_control_external]):
        ax[d].boxplot(v,
                      positions=[i],
                      widths=.9,
                      showfliers=False,
                      boxprops={"color":f"C{i}","linewidth":2},
                      meanprops={"color":"black"},
                      medianprops={"color":f"C{i}","linewidth":2},
                      capprops={"linewidth":2},
                      whiskerprops={"linewidth":2})
    ax[d].set_title(dataset.replace("_"," ").upper(), fontsize=8, fontweight="bold", loc="left")
    ax[d].set_ylabel("Cosine Similarity", fontsize=6, fontweight="bold")
    ax[d].set_xticks(list(range(i+1)))
    ax[d].set_xticklabels(["Depression\n(Within)","Control\n(Within)","Depression\n(Across)","Control\n(Across)"],
                          fontsize=6)
ax[-1].axis("off")
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}cosine_similarity_depression_boxplot_GloVe.png", dpi=150)
plt.close(fig)

########################
### GloVe Aggregated Similarity
########################

LOGGER.info("Examining Aggregated GloVe Cosine Similarity")

## Aggregate Counts by Data Set and Class
X_agg = []
groups = []
for dataset in DATA_SETS:
    for t, target in enumerate(["Control","Depression"]):
        mask = np.nonzero(np.logical_and(y_dataset == dataset,
                                         y_depression==t))[0]
        X_agg.append(np.array((X[mask]>0).sum(axis=0)))
        groups.append((dataset.replace("_"," ").upper(), target))
X_agg = np.vstack(X_agg)

## Get Transformed
X_agg_glove = glove.transform(X_agg)
X_agg_glove = normalize(X_agg_glove, axis=1, norm="l2")

## Compute Pairwise Similarity
X_agg_sim = pairwise.cosine_similarity(X_agg_glove)

## Add Null to Upper Triangle
mask = np.ones_like(X_agg_sim)
mask[np.triu_indices_from(mask)] = np.nan
mask[np.diag_indices_from(mask)] = 1
X_agg_sim = X_agg_sim * mask

## Plot Heatmap
fig, ax = plt.subplots()
m = ax.imshow(X_agg_sim,
              cmap=plt.cm.Purples,
              aspect="auto",
              interpolation="nearest")
for i, row in enumerate(X_agg_sim):
    for j, val in enumerate(row):
        if pd.isnull(val):
            continue
        ax.text(j, i, "{:.4f}".format(val), fontsize=5, ha="center",va="center")
cbar = fig.colorbar(m)
cbar.set_label("Cosine Similarity", fontweight="bold", fontsize=8)
ax.set_xticks(list(range(len(groups))))
ax.set_xticklabels(["{}\n({})".format(i[0],i[1]) for i in groups], fontsize=6, rotation=45, ha="right")
ax.set_yticks(list(range(len(groups))))
ax.set_yticklabels(["{}\n({})".format(i[0],i[1]) for i in groups], fontsize=6)
ax.set_ylim(len(groups)-.5, -.5)
ax.set_title("Aggregate GloVe Cosine Similarity", fontsize=10, loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}aggregate_GloVe_cosine_similarity.png", dpi=150)
plt.close()

########################
### LIWC Distributions
########################

"""
Idea: Use Uncertainty Propagation to Understand Differences
Source: http://ipl.physics.harvard.edu/wp-uploads/2013/03/PS3_Error_Propagation_sp13.pdf
"""

LOGGER.info("Plotting Class Distributional Differences of LIWC Features")

## Bootstrap Parameters
boot_frac = 0.7
boot_samples = 100

## Store Results
liwc_stats = []

## Compute and Plot Distributional Differences Between Classes
fig, ax = plt.subplots(1, 5, figsize=(10,4.5))
k_top = 10
data_order = [i for i in DS_ORDER if i in set(y_dataset)]
for d, dataset in enumerate(data_order):
    ## Identify Masks
    control_mask = np.nonzero(np.logical_and(y_dataset[nn_glove_map] == dataset,
                                             y_depression[nn_glove_map]==0))[0]
    depression_mask = np.nonzero(np.logical_and(y_dataset[nn_glove_map] == dataset,
                                                y_depression[nn_glove_map]==1))[0]
    ## Bootstrap Sample The Difference
    con_liwc_normed = X_liwc_normed[nn_glove_map][control_mask]
    dep_liwc_normed = X_liwc_normed[nn_glove_map][depression_mask]
    cache = []
    for _ in range(boot_samples):
        ## Sample
        con_samp = np.random.choice(con_liwc_normed.shape[0],
                                    size=int(con_liwc_normed.shape[0]*boot_frac),
                                    replace=True)
        dep_samp = np.random.choice(dep_liwc_normed.shape[0],
                                    size=int(dep_liwc_normed.shape[0]*boot_frac),
                                    replace=True)
        ## Compute Mean Difference
        con_mean = np.nanmean(con_liwc_normed[con_samp],axis=0)
        dep_mean = np.nanmean(dep_liwc_normed[dep_samp],axis=0)
        cache.append(dep_mean-con_mean)
    cache = np.vstack(cache)
    cache_range = np.nanpercentile(cache, q=[2.5,50,97.5],axis=0)
    ## Format and Sort
    uncertainty = pd.DataFrame(cache_range.T,
                               index=liwc.names,
                               columns=["lower","median","upper"])
    uncertainty = uncertainty.sort_values("median",ascending=True)
    ## Cache
    to_cache = uncertainty[["median"]].copy()
    to_cache.rename(columns={"median":dataset},inplace=True)
    liwc_stats.append(to_cache)
    ## Plot Top
    top = uncertainty.head(k_top).append(uncertainty.tail(k_top))
    ax[d].barh(list(range(len(top))),
               top["upper"]-top["lower"],
               left=top["lower"],
               alpha=0.5,
               color=["darkred"]*k_top+["navy"]*k_top,
    )
    ax[d].scatter(top["median"],
                  list(range(len(top))),
                  color=["darkred"]*k_top+["navy"]*k_top,
                  alpha=.9,
                  zorder=10)
    ax[d].axvline(0, color="black", linestyle="--", alpha=0.5, zorder=-1)
    ax[d].set_yticks(list(range(len(top))))
    ax[d].set_yticklabels(top.index.tolist(), fontsize=10)
    ax[d].set_title(DS_NAMES[dataset], fontsize=14, fontweight="bold", loc="center")
    ax[d].set_ylim(-.5,len(top)-.5)
ax[0].set_ylabel("LIWC Dimension", fontsize=14, fontweight="bold")
ax[2].set_xlabel("Prevalence\n(Depression-Control)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.subplots_adjust(wspace=0.75)
plt.savefig(f"{PLOT_DIR}distributional_differences_LIWC.png", dpi=300)
plt.close()

## Format and Cache Stat Comparison
liwc_stats = pd.concat(liwc_stats,axis=1,sort=True)
liwc_stats = liwc_stats.loc[liwc_stats.mean(axis=1).sort_values().index].copy()
liwc_stats.to_csv(f"{RESULTS_DIR}liwc_class_prevalence.csv")

## Plot Comparison
liwc_rank = liwc_stats.rank(axis=0)
most_volatile = liwc_rank.std(axis=1).sort_values().index[-20:]
plot_data = liwc_rank.loc[most_volatile].T
fig, ax = plt.subplots(figsize=(10,5.8))
m = ax.imshow(plot_data,
              cmap=plt.cm.Blues,
              aspect="auto",
              interpolation=None,
              alpha=.7)
cbar = fig.colorbar(m)
cbar.set_label("Association Strength\nWith Depression (Rank)",
               fontsize=10,
               fontweight="bold")
for i, row in enumerate(plot_data.values):
    for j, val in enumerate(row):
        ax.text(j,i, int(val), ha="center", va="center", color="black")
ax.set_yticks(list(range(plot_data.shape[0])))
ax.set_yticklabels([DS_NAMES[i] for i in plot_data.index])
ax.set_ylim(4.5,-.5)
ax.set_xticks(list(range(plot_data.shape[1])))
ax.set_xticklabels(plot_data.columns.tolist(),rotation=45,ha="right")
ax.set_xlabel("LIWC Dimension", fontsize=10, fontweight="bold")
ax.set_title("LIWC Dimensions With Highest Cross-domain Variance",
             loc="left",
             fontsize=12,
             fontweight="bold")
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}volatile_liwc_categories.png", dpi=300)
plt.close()
