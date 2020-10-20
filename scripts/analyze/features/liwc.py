
## Directories
PLOT_DIR = "./plots/liwc/"
DATA_DIR = "./data/processed/"
CACHE_DIR = "./data/results/cache/liwc/"

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
    col_subset = ["user_id_str","depression","gender","age","split","num_posts","num_words","dataset","source"]
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

## Label Mappings
label_maps = {}
for dataset in DATA_SETS:
    dataset_subset = datasets_df.loc[(datasets_df["dataset"]==dataset)]
    label_maps[dataset] = dataset_subset.set_index("source")["depression"].to_dict()

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
depression_filemap = datasets_df.set_index("source")["depression"].to_dict()
y_dataset = list(map(lambda f: dataset_filemap[f], filenames))
y_depression = list(map(lambda f: depression_filemap[f], filenames))

## Filtering Function
def create_subset(X,
                  y_dataset,
                  y_depression,
                  data_classes=list(DATA_SETS.keys()),
                  dep_classes=["depression"],
                  split_ds=True):
    """

    """
    ## Subset
    dep_ind = [i for i, c in enumerate(y_depression) if c in dep_classes]
    data_ind = [i for i, c in enumerate(y_dataset) if c in data_classes]
    subset_ind = sorted(set(dep_ind)&set(data_ind))
    ## Y Formulation
    y_dep_subset = [y_depression[i] for i in subset_ind]
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
### Experimment 1: LIWC Stability
########################

LOGGER.info("Starting Experiment 1: Stablility of LIWC Coefficients")

## Parameters
SAMPLE_SIZE = .7
SAMPLES = 50

## Coefficient Cache
coefs = {ds:[] for ds in DATA_SETS.keys()}
np.random.seed(42)
for train_dataset in DATA_SETS.keys():
    LOGGER.info(f"Finding LIWC Coefficients for {train_dataset}")
    ## Filenames
    train_samples = label_maps[train_dataset]
    ## Initialize Vectorizers
    train_f2v = File2Vec(vocab_kwargs=vocab_kwargs)
    ## Learn Vocabularies
    train_f2v.vocab = train_f2v.vocab.fit(sorted(train_samples),
                                          chunksize=VOCAB_CHUNKSIZE,
                                          jobs=NUM_JOBS,
                                          min_date=DATA_SETS[train_dataset]["date_boundaries"].get("min_date"),
                                          max_date=DATA_SETS[train_dataset]["date_boundaries"].get("max_date"))
    ## Initialize Vectorizers
    train_f2v._initialize_dict_vectorizer()
    ## Vectorize
    filenames_train, X_train = train_f2v._vectorize_files(sorted(train_samples), 8,
                                          min_date=DATA_SETS[train_dataset]["date_boundaries"].get("min_date"),
                                          max_date=DATA_SETS[train_dataset]["date_boundaries"].get("max_date"))
    ## LIWC Representation
    liwc_train = LIWCTransformer(train_f2v.vocab)
    ## Get Representations
    X_train = liwc_train.fit_transform(X_train)
    ## Labels
    y_train = train_f2v._vectorize_labels(filenames_train, train_samples)
    ## Cycle Through Samples
    for _ in range(SAMPLES):
        ## Select Sample
        sample_ind = np.random.choice(X_train.shape[0],
                                      size = int(X_train.shape[0]*SAMPLE_SIZE),
                                      replace=False)
        X_train_sample = X_train[sample_ind]
        y_train_sample = y_train[sample_ind]
        ## Scaling
        scaler = StandardScaler()
        X_train_sample = scaler.fit_transform(X_train_sample)
        ## Fit Model
        model = LogisticRegression(C=1, solver="lbfgs", random_state=42)
        model = model.fit(X_train_sample, y_train_sample)
        ## Cache Coefficients
        coefs[train_dataset].append(model.coef_[0])

## Format
coef_df = []
for dataset, co in coefs.items():
    co_stack = pd.DataFrame(np.vstack(co), columns=liwc_train.names)
    co_stack["dataset"] = dataset
    coef_df.append(co_stack)
coef_df = pd.concat(coef_df)

## Cache
coef_df.to_csv(f"{CACHE_DIR}liwc_stability_coefs.csv", index=False)

## Aggregate
agg_coef_df_mean = coef_df.groupby(["dataset"]).mean()
agg_coef_df_std = coef_df.groupby(["dataset"]).std()

## Pair Plot
datasets = agg_coef_df_mean.index.tolist()
fig, ax = plt.subplots(len(datasets), len(datasets),figsize=(10,5.8))
for i, d1 in enumerate(datasets):
    for j, d2 in enumerate(datasets):
        if j > i:
            ax[i, j].axis("off")
            continue
        if j == i:
            ax[i, j].hist(agg_coef_df_mean.loc[d2],
                          alpha=.5,
                          bins=15)
        else:
            ax[i, j].errorbar(agg_coef_df_mean.loc[d2],
                              agg_coef_df_mean.loc[d1],
                              xerr=agg_coef_df_std.loc[d2],
                              yerr=agg_coef_df_std.loc[d1],
                               fmt="o",
                              ecolor="black",
                              alpha = .5,
                              ms = 2.5)
            corr = spearmanr(agg_coef_df_mean.loc[d2], agg_coef_df_mean.loc[d1]).correlation
            ax[i, j].set_title("Correlation: {:.3f}".format(corr), loc="left", fontsize=8)
        if j == 0:
            ax[i, j].set_ylabel(d1)
        if i == len(datasets) - 1:
            ax[i, j].set_xlabel(d2)
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}liwc_coefficient_correlation.png")
plt.close()

########################
### Experimment 2: LIWC's Base Vocabulary
########################

"""
Idea: We currently restrict the LIWC transformation to the vocabulary of the
training domain. However, since LIWC is an external resource, there actually
isn't a strict need to do this (e.g. we can learn separate vocabularies and
transform the feature set into the same LIWC space). This applies as well for
averaged embeddings. The hypothesis is that models that don't abide by this vocabulary
restriction outperform those who do.
"""

def run_liwc_experiment(label_maps,
                        train_dataset,
                        test_dataset,
                        fix_vocabulary=False,
                        scale_independent=False,
                        verbose=False):
    """

    """
    ## Filenames
    train_samples = label_maps[train_dataset]
    test_samples = label_maps[test_dataset]
    ## Vectorize
    if not fix_vocabulary:
        ## Initialize Vectorizers
        train_f2v = File2Vec(vocab_kwargs=vocab_kwargs)
        test_f2v = File2Vec(vocab_kwargs=vocab_kwargs)
        ## Learn Vocabularies
        train_f2v.vocab = train_f2v.vocab.fit(sorted(train_samples),
                                              chunksize=VOCAB_CHUNKSIZE,
                                              jobs=NUM_JOBS,
                                              min_date=DATA_SETS[train_dataset]["date_boundaries"].get("min_date"),
                                              max_date=DATA_SETS[train_dataset]["date_boundaries"].get("max_date"))
        test_f2v.vocab = test_f2v.vocab.fit(sorted(test_samples),
                                            chunksize=VOCAB_CHUNKSIZE,
                                            jobs=NUM_JOBS,
                                            min_date=DATA_SETS[test_dataset]["date_boundaries"].get("min_date"),
                                            max_date=DATA_SETS[test_dataset]["date_boundaries"].get("max_date"))
        ## Initialize Vectorizers
        train_f2v._initialize_dict_vectorizer()
        test_f2v._initialize_dict_vectorizer()
        ## Vectorize
        filenames_train, X_train = train_f2v._vectorize_files(sorted(train_samples), 8,
                                                              min_date=DATA_SETS[train_dataset]["date_boundaries"].get("min_date"),
                                                              max_date=DATA_SETS[train_dataset]["date_boundaries"].get("max_date"))
        filenames_test, X_test = test_f2v._vectorize_files(sorted(test_samples), 8,
                                                           min_date=DATA_SETS[test_dataset]["date_boundaries"].get("min_date"),
                                                           max_date=DATA_SETS[test_dataset]["date_boundaries"].get("max_date"))
        ## LIWC Representation
        liwc_train = LIWCTransformer(train_f2v.vocab)
        liwc_test = LIWCTransformer(test_f2v.vocab)
        ## Get Representations
        X_train = liwc_train.fit_transform(X_train)
        X_test = liwc_test.fit_transform(X_test)
    else:
        ## Initialize Vectorizers
        train_f2v = File2Vec(vocab_kwargs=vocab_kwargs)
        ## Learn Vocabularies
        train_f2v.vocab = train_f2v.vocab.fit(sorted(train_samples),
                                              chunksize=VOCAB_CHUNKSIZE,
                                              jobs=NUM_JOBS,
                                              min_date=DATA_SETS[train_dataset]["date_boundaries"].get("min_date"),
                                              max_date=DATA_SETS[train_dataset]["date_boundaries"].get("max_date"))
        ## Initialize Vectorizers
        train_f2v._initialize_dict_vectorizer()
        ## Vectorize
        filenames_train, X_train = train_f2v._vectorize_files(sorted(train_samples), 8,
                                                              min_date=DATA_SETS[train_dataset]["date_boundaries"].get("min_date"),
                                                              max_date=DATA_SETS[train_dataset]["date_boundaries"].get("max_date"))
        filenames_test, X_test = train_f2v._vectorize_files(sorted(test_samples), 8,
                                                            min_date=DATA_SETS[test_dataset]["date_boundaries"].get("min_date"),
                                                            max_date=DATA_SETS[test_dataset]["date_boundaries"].get("max_date"))
        ## LIWC Representation
        liwc_train = LIWCTransformer(train_f2v.vocab)
        ## Get Representations
        X_train = liwc_train.fit_transform(X_train)
        X_test = liwc_train.fit_transform(X_test)
    ## Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if scale_independent:
        X_test = scaler.fit_transform(X_test)
    else:
        X_test = scaler.transform(X_test)
    ## Labels
    y_train = train_f2v._vectorize_labels(filenames_train, train_samples, pos_class="depression")
    y_test = train_f2v._vectorize_labels(filenames_test, test_samples, pos_class="depression")
    ## Fit Model
    model = LogisticRegression(C=1, solver="lbfgs")
    model = model.fit(X_train, y_train)
    ## Make Predictions
    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]
    ## Evaluate Performance
    score_train = pd.DataFrame(metrics.classification_report(y_train, y_train_pred_prob>0.5,output_dict=True))
    score_test = pd.DataFrame(metrics.classification_report(y_test, y_test_pred_prob>0.5,output_dict=True))
    ## Output
    if verbose:
        print("~"*50 + f"\nTrain Performance ({train_dataset})\n" + "~"*50)
        print(score_train)
        print("~"*50 + f"\nTest Performance ({test_dataset})\n" + "~"*50)
        print(score_test)
    return score_train, score_test

def get_stats(classification_report):
    """

    """
    stats = {"accuracy":classification_report["accuracy"].values[0],
             "precision":classification_report.loc["precision"][1],
             "recall":classification_report.loc["recall"][1],
             "f1":classification_report.loc["f1-score"][1],
             "weighted_precision":classification_report.loc["precision"]["weighted avg"],
             "weighted_recall":classification_report.loc["recall"]["weighted avg"],
             "weighted_f1":classification_report.loc["f1-score"]["weighted avg"]}
    return stats

LOGGER.info("Starting Experiment 2: LIWC Vocabulary vs. Restricted LIWC Vocabulary")

## Run Test(s)
results = []
datasets = sorted(DATA_SETS.keys())
for source_dataset in datasets:
    for target_dataset in datasets:
        if source_dataset == target_dataset:
            continue
        LOGGER.info(f"Starting LIWC Test w/ Source: {source_dataset}, Target: {target_dataset}")
        for fixed_vocab in [False, True]:
            for scale_independent in [False, True]:
                score_train, score_test = run_liwc_experiment(label_maps,
                                                              source_dataset,
                                                              target_dataset,
                                                              fix_vocabulary=fixed_vocab,
                                                              scale_independent=scale_independent)
                stats_train = get_stats(score_test)
                stats_test = get_stats(score_test)
                for d in [stats_train, stats_test]:
                    d["fixed_vocabulary"] = fixed_vocab
                    d["scale_independent"] = scale_independent
                    d["source"] = source_dataset
                    d["target"] = target_dataset
                stats_train["group"] = "train"
                stats_test["group"] = "test"
                results.append(stats_train)
                results.append(stats_test)

## Concatenate Results
results = pd.DataFrame(results)

## Cache
results.to_csv(f"{CACHE_DIR}liwc_base_vocabulary.csv", index=False)

## Create Analysis Plot
def plot_fv_scaling_comparison(results,
                               metric):
    """

    """
    datasets = results["source"].unique()
    fig, ax = plt.subplots(2, 3, figsize=(10,5.8), sharey=True)
    ax = flatten(ax)
    bar_width = .95 / 4
    for d, ds in enumerate(datasets):
        d_ind = 0
        xticks = []
        xticklabels = []
        for d2, ds2 in enumerate(datasets):
            if ds == ds2:
                continue
            xticks.append(d_ind + .5)
            xticklabels.append(ds2)
            t_ind = 0
            for f, fv in enumerate([False, True]):
                for s, si in enumerate([False, True]):
                    p_data = results.loc[(results["source"]==ds)&
                                        (results["target"]==ds2)&
                                        (results["fixed_vocabulary"]==fv)&
                                        (results["scale_independent"]==si)&
                                        (results["group"]=="test")].iloc[0]
                    ax[d].bar(0.025 + d_ind + t_ind*bar_width,
                            p_data[metric],
                            width = bar_width,
                            label = f"FV: {fv}, IndScale: {si}" if d_ind == 0 else "",
                            align = "edge",
                            color=f"C{f}",
                            alpha=.8 if s == 0 else .5)
                    t_ind += 1
            d_ind += 1
        ax[d].set_xticks(xticks)
        ax[d].set_xticklabels(xticklabels)
        ax[d].set_title(f"Source: {ds}")
        ax[d].set_ylabel(metric)
    ax[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.2)
    return fig, ax

for metric in ["accuracy","precision","recall","weighted_precision","weighted_recall","weighted_f1"]:
    fig, ax = plot_fv_scaling_comparison(results, metric)
    fig.savefig(f"{PLOT_DIR}liwc_vocab_scaling_{metric}.png")
    plt.close()
