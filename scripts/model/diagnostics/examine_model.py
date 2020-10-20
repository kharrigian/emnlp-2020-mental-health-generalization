
##########################
### Imports
##########################

## Standard Libary
import os
import json
import argparse

## External Libaries
import joblib
import demoji
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Local Modules
from mhlib.util.logging import initialize_logger

## Create Logger
LOGGER = initialize_logger()

##########################
### Functions
##########################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.

    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Run modeling experiments")
    ## Generic Arguments
    parser.add_argument("model_filepath",
                        type=str,
                        help="Path to your model joblib file")
    parser.add_argument("output_dir",
                        type=str,
                        help="Where to create output folder")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.model_filepath):
        raise ValueError(f"Could not find model file {args.model_filepath}")
    return args

def replace_emojis(features):
    """

    """
    features_clean = []
    for f in features:
        f_res = demoji.findall(f)
        if len(f_res) > 0:
            for x,y in f_res.items():
                f = f.replace(x,f"EMOJI={y}")
            features_clean.append(f)
        else:
            features_clean.append(f)
    return features_clean

def identify_feature_indices(model):
    """

    """
    ## Flags
    feature_flags = model.preprocessor._feature_flags
    ## Feature Names
    features = model.get_feature_names()
    ## Indices
    feature_indices = {}
    ## Parse Features
    if "bag_of_words" in feature_flags and feature_flags.get("bag_of_words"):
        feature_indices["bag_of_words"] = [i for i, f in enumerate(features) if isinstance(f, tuple)]
    if "tfidf" in feature_flags and feature_flags.get("tfidf"):
        feature_indices["tfidf"] = [i for i, f in enumerate(features) if isinstance(f, tuple)]
    if "lda" in feature_flags and feature_flags.get("lda"):
        feature_indices["lda"] = [i for i, f in enumerate(features) if isinstance(f, str) and f.startswith("LDA_TOPIC_")]
    if "liwc" in feature_flags and feature_flags.get("liwc"):
        feature_indices["liwc"] = [i for i, f in enumerate(features) if isinstance(f, str) and f.startswith("LIWC=")]
    if "glove" in feature_flags and feature_flags.get("glove"):
        feature_indices["glove"] = [i for i, f in enumerate(features) if isinstance(f, str) and f.startswith("GloVe_Dim_")]
    return feature_indices

def plot_feature_bar_plot(coefs,
                          k_top=20):
    """

    """
    ## Isolate Top Features
    features = coefs.head(k_top).index.tolist() + ["..."] + coefs.tail(k_top).index.tolist()
    features = replace_emojis(features)
    values = coefs.head(k_top).values.tolist() + [0] + coefs.tail(k_top).values.tolist()
    index = list(range(k_top*2 + 1))
    ## Create Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.barh(index,
            values,
            alpha=0.5,
            color="C0")
    ax.axvline(0, color = "black", linestyle="--", alpha=.5)
    ax.set_yticks(index)
    ax.set_yticklabels(features, fontsize=6)
    ax.set_xlabel("Coefficient")
    ax.set_ylim(-.5, k_top*2+.5)
    fig.tight_layout()
    return fig, ax

def interpret_ngram_features(model, indices):
    """

    """
    ## Series
    coefs = pd.Series(index=model.get_feature_names(), data=model.model.coef_[0]) 
    coefs = coefs.iloc[indices]
    ## Sort
    coefs.sort_values(inplace=True)
    ## Format
    coefs.index = coefs.index.map(lambda i: "_".join(i))
    ## Create Plot
    fig, ax = plot_feature_bar_plot(coefs)
    return fig, ax

def interpret_lda_features(model, indices, k_reps=10):
    """

    """
    ## Series
    coefs = pd.Series(index=model.get_feature_names(), data=model.model.coef_[0]) 
    coefs = coefs.iloc[indices]
    ## Unigrams
    ngrams = model.vocab.get_ordered_vocabulary()
    ## LDA Model
    representative_features = []
    lda = model.preprocessor._transformers["lda"]
    for c, component in enumerate(lda.components_):
        top_ngrams = np.argsort(component)[::-1][:k_reps]
        top_ngrams = ", ".join(["_".join(ngrams[i]) for i in top_ngrams])
        representative_features.append(f"{c+1}) {top_ngrams}")
    ## Update Index
    coefs.index = representative_features
    ## Sort
    coefs.sort_values(inplace=True)
    ## Create Plot
    fig, ax = plot_feature_bar_plot(coefs)
    return fig, ax

def interpret_liwc_features(model, indices):
    """

    """
    ## Series
    coefs = pd.Series(index=model.get_feature_names(), data=model.model.coef_[0]) 
    coefs = coefs.iloc[indices]
    ## Sort
    coefs.sort_values(inplace=True)
    ## Create Plot
    fig, ax = plot_feature_bar_plot(coefs)
    return fig, ax

def interpret_glove_features(model, indices, k_reps=5):
    """

    """
    ## Series
    coefs = pd.Series(index=model.get_feature_names(), data=model.model.coef_[0]) 
    coefs = coefs.iloc[indices]
    ## Sort
    coefs.sort_values(inplace=True)
    ## Create Plot
    fig, ax = plot_feature_bar_plot(coefs)
    return fig, ax

def main():
    """

    """
    ## Parse Arguments
    args = parse_arguments()
    ## Load Model
    LOGGER.info(f"Loading Model: {args.model_filepath}")
    model = joblib.load(args.model_filepath)
    ## Identify Feature Indices
    LOGGER.info("Parsing Feature Indices")
    feature_indices = identify_feature_indices(model)
    ## Create Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ## Create Plots
    if "bag_of_words" in feature_indices:
        LOGGER.info("Interpreting N-Gram Features")
        fig, ax = interpret_ngram_features(model, feature_indices["bag_of_words"])
        fig.savefig(f"{args.output_dir}bag_of_words.png", dpi=150)
        plt.close(fig)
    if "tfidf" in feature_indices:
        LOGGER.info("Interpreting N-Gram Features")
        fig, ax = interpret_ngram_features(model, feature_indices["tfidf"])
        fig.savefig(f"{args.output_dir}tfidf.png", dpi=150)
        plt.close(fig)
    if "liwc" in feature_indices:
        LOGGER.info("Interpreting LIWC Features")
        fig, ax = interpret_liwc_features(model, feature_indices["liwc"])
        fig.savefig(f"{args.output_dir}liwc.png", dpi=150)
        plt.close(fig)
    if "glove" in feature_indices:
        LOGGER.info("Interpreting GloVe Features")
        fig, ax = interpret_glove_features(model, feature_indices["glove"])
        fig.savefig(f"{args.output_dir}glove.png", dpi=150)
        plt.close(fig)
    if "lda" in feature_indices:
        LOGGER.info("Interpreting LDA Features")
        fig, ax = interpret_lda_features(model, feature_indices["lda"])
        fig.savefig(f"{args.output_dir}lda.png", dpi=150)
        plt.close(fig)

##########################
### Execution
##########################

if __name__ == "__main__":
    _ = main()