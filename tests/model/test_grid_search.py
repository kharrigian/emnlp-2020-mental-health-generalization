#######################
### Imports
#######################

## Standard Library
import os
import json
import gzip
from datetime import datetime

## External Libraries
import pytest
import numpy as np
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score

## Local Modules
from mhlib.preprocess.tokenizer import Tokenizer
from mhlib.model.model import MentalHealthClassifier
from mhlib.model.grid_search import run_grid_search

#######################
### Fixtures
#######################

@pytest.fixture(scope="module")
def basic_hyperparameter_config():
    """

    """
    ## Path
    config_path = os.path.dirname(__file__) + "/../data/basic.json"
    ##
    config = \
            [
            {
                "name":"_default",
                "feature_selector_ablation":True,
                "feature_set_ablation":True,
                "model":[
                        {
                        "name":"logistic",
                        "solver":["lbfgs"],
                        "max_iter":[1000],
                        "C":[1,10],
                        "class_weight":[None],
                        "random_state":[42]
                        }
                    ],
                "features":{
                    "bag_of_words":{},
                    "tfidf":{},
                    "glove":{
                        "dim":[25],
                        "pooling":["mean"]
                    },
                    "liwc":{
                        "norm":["max"]
                    },
                    "lda":{
                        "n_components":[25],
                        "doc_topic_prior":[None],
                        "topic_word_prior":[None],
                        "random_state":[42]
                    }
                },
                "standardize":[True],
                "feature_selector":{
                    "kldivergence":{
                        "top_k":[50, 100],
                        "min_support":[10],
                        "add_lambda":[0.01],
                        "beta":[0.1]
                    }
                }
            }
        ]
    ## Dump
    with open(config_path, "w") as the_file:
        json.dump(config, the_file)
    yield config_path
    ## Teardown
    _ = os.system(f"rm {config_path}")

@pytest.fixture(scope="module")
def newsgroups_data():
    """

    """
    ## Cache Path
    base_path = os.path.dirname(__file__) + "/../data/"
    ## Get Newsgroups Data (Subset by Categories)
    data = fetch_20newsgroups(data_home=base_path,
                              subset="train",
                              categories=["rec.sport.hockey",
                                          "sci.med"],
                              shuffle=False,
                              remove=("headers","footers","quotes"),
                              download_if_missing = True)
    ## Processed Documents Cache
    processed_data_folder = f"{base_path}newsgroups_processed/"
    if not os.path.exists(processed_data_folder):
        os.mkdir(processed_data_folder)
    ## Initialize Tokenizer
    tokenizer = Tokenizer(stopwords=set(),
                          keep_case=False,
                          negate_handling=True,
                          negate_token=True,
                          upper_flag=True,
                          keep_punctuation=False,
                          keep_numbers=False,
                          expand_contractions=True,
                          keep_user_mentions=False,
                          keep_pronouns=True,
                          keep_url=False,
                          keep_hashtags=False,
                          keep_retweets=False,
                          emoji_handling=None)
    processing_time = int(datetime.utcnow().timestamp())
    label_dict = {}
    for t, (text, target) in enumerate(zip(data["data"][:100], data["target"][:100])):
        target_name = "depression" if data["target_names"][target] == "rec.sport.hockey" else "control"
        dpoint_processed = [{
            "text":text,
            "text_tokenized":tokenizer.tokenize(text),
            "created_utc":1,
            "date_processed_utc":processing_time,
            "label":target_name,
            "id_str":t,
        }]
        t_file = f"{processed_data_folder}document_{t}.json.tar.gz"
        with gzip.open(t_file, "wt", encoding="utf-8") as the_file:
            json.dump(dpoint_processed, the_file)
        label_dict[t_file] = target_name
    data_files = sorted(label_dict.keys())
    yield data_files, label_dict
    ## Teardown
    _ = os.system(f"rm -rf {processed_data_folder}")

#######################
### Debugging
#######################

def test_grid_search(basic_hyperparameter_config,
                     newsgroups_data):
    """

    """
    ## Get Hyperparameter Grid
    hyperparameter_config = basic_hyperparameter_config
    ## Get Data 
    data_files, label_dict = newsgroups_data
    ## Initialize Model
    model = MentalHealthClassifier(target_disorder="depression")
    ## Run Grid Search
    grid_search_results, grid_search_best_result, grid_search_best_score = run_grid_search(model=model,
                                train_files=data_files,
                                train_label_dict=label_dict,
                                dev_files=None,
                                dev_label_dict=None,
                                config=hyperparameter_config,
                                random_state=42,
                                test_size=.2,
                                dev_in_vocab=False,
                                score_func=f1_score)
    ## Tests
    assert isinstance(grid_search_results, list)
    assert isinstance(grid_search_best_result, dict)
    assert isinstance(grid_search_best_score, float)
    assert len(grid_search_results) == 30
    sorted_res = sorted(grid_search_results, key = lambda x: x["dev_score"], reverse=True)
    assert sorted_res[0]["dev_score"] == grid_search_best_score
    assert sorted_res[0]["dev_score"] - sorted_res[-1]["dev_score"] > 0.2
