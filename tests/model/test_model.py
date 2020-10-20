
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
from sklearn.metrics import accuracy_score

## Local Modules
from mhlib.model.vocab import Vocabulary
from mhlib.preprocess.tokenizer import Tokenizer
from mhlib.model.model import MentalHealthClassifier
from mhlib.model.feature_extractors import FeaturePreprocessor
from mhlib.model.file_vectorizer import File2Vec

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
                "feature_selector_ablation":False,
                "feature_set_ablation":False,
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
### Tests
#######################

def test_MentalHealthClassifier_repr():
    """

    """
    ## Initialize Classifier
    preprocessing_kwargs = {"feature_flags":{"tfidf":True},"feature_kwargs":{"tfidf":{}}}
    model = MentalHealthClassifier(target_disorder="depression",
                                   model="logistic",
                                   vocab_kwargs={"preserve_case":False},
                                   preprocessing_kwargs=preprocessing_kwargs)
    ## Get REPR
    model_repr = model.__repr__()
    ## Tests
    assert model_repr.startswith("MentalHealthClassifier(")

def test_MentalHealthClassifier_learn_vocabulary(newsgroups_data):
    """

    """
    ## Initialize Model
    model = MentalHealthClassifier(target_disorder="depression")
    ## Learn Vocabulary
    test_files, _ = newsgroups_data
    model._learn_vocabulary(test_files)
    ## Tests
    assert hasattr(model, "vocab")
    assert hasattr(model, "_count2vec")
    assert isinstance(model.vocab, Vocabulary)

def test_MentalHealthClassifier_fit_with_grid_search(basic_hyperparameter_config,
                                                     newsgroups_data):
    """

    """
    ## Parse Fixtures
    config_file = basic_hyperparameter_config
    test_files, label_dict = newsgroups_data
    ## Initialize Model
    model = MentalHealthClassifier(target_disorder="depression")
    ## Fit Model with Grid Search
    model = model.fit_with_grid_search(test_files,
                                       label_dict,
                                       dev_files=None,
                                       dev_label_dict=None,
                                       config=config_file,
                                       test_size=.2,
                                       dev_in_vocab=False,
                                       cache_results=True,
                                       return_training_preds=False)
    ## Check Results
    assert model._grid_search_results is not None
    assert model._grid_search_best_result is not None
    assert model._grid_search_best_score is not None
    assert "n_iter_" in model.model.__getstate__()
    assert len(model._grid_search_results) == 4


def test_MentalHealthClassifier_get_feature_names(newsgroups_data):
    """

    """
    ## Parse Fixture
    test_files, label_dict = newsgroups_data
    ## Initialize Model
    preprocessing_kwargs = {"feature_flags":{"tfidf":True, "glove":True, "liwc":True,"lda":True},
                            "feature_kwargs":{"tfidf":{},"glove":{},"liwc":{},"lda":{}}}
    model = MentalHealthClassifier(target_disorder="depression",
                                   preprocessing_kwargs=preprocessing_kwargs)
    ## Learn Vocabulary
    model._learn_vocabulary(test_files)
    ## Get X
    filenames, X = model._vectorize_files(test_files)
    ## Fit Preprocessor
    model.preprocessor = FeaturePreprocessor(model.vocab,
                                             model._preprocessing_kwargs.get("feature_flags"),
                                             model._preprocessing_kwargs.get("feature_kwargs"),
                                             False)
    X_T = model.preprocessor.fit_transform(X)
    ## Get Feature Names
    feature_names = model.get_feature_names()
    ## Tests
    assert isinstance(feature_names, list)
    assert len(feature_names) == X_T.shape[1]
    assert len([i for i in feature_names if isinstance(i, str) and i.startswith("GloVe")]) == \
           model.preprocessor._transformers["glove"].dim
    assert len([i for i in feature_names if isinstance(i, str) and i.startswith("LIWC=")]) == 64
    assert len([i for i in feature_names if isinstance(i, tuple)]) == len(model.vocab.vocab)
    assert len([i for i in feature_names if isinstance(i, str) and i.startswith("LDA_TOPIC_")]) == \
           model.preprocessor._transformers["lda"].n_components
    
def test_MentalHealthClassifier_load_vectors(newsgroups_data):
    """

    """
    ## Parse Fixture
    test_files, label_dict = newsgroups_data
    ## Initialize Model
    model = MentalHealthClassifier(target_disorder="depression")
    ## Learn Vocabulary
    model._learn_vocabulary(test_files)
    ## Vectorize
    filenames, X, y = model._load_vectors(test_files, label_dict)
    ## Test
    assert X.shape[0] == len(label_dict)
    assert X.shape[1] == len(model.vocab.vocab)
    assert X.shape[0] == y.shape[0] == len(filenames)
    for f, y_f in zip(filenames, y):
        assert {"depression":1,"control":0}[label_dict[f]] == y_f
    
def test_MentalHealthClassifier_fit(newsgroups_data):
    """

    """
    ## Parse Fixture
    test_files, label_dict = newsgroups_data
    ## Initialize Model
    model = MentalHealthClassifier(target_disorder="depression")
    ## Fit Model
    model = model.fit(test_files, label_dict, False)
    ## Tests
    assert hasattr(model, "preprocessor")
    assert hasattr(model, "selector")
    assert max(model.model.__getstate__()["n_iter_"]) > 0
    ## Fit Model Again, Getting Training probability predictions
    model, y_pred = model.fit(test_files, label_dict, True)
    assert isinstance(y_pred, dict)
    assert np.all([isinstance(i, float) for i in y_pred.values()])
    assert len(y_pred) == len(label_dict)
    ## Fit Model Again, This Time with a non-probabalistic classifier
    model = MentalHealthClassifier("depression", "svm", vocab_kwargs={"max_vocab_size":10})
    model, y_pred = model.fit(test_files, label_dict, True)
    assert isinstance(y_pred, dict)
    assert np.all([isinstance(i, float) for i in y_pred.values()])
    assert len(y_pred) == len(label_dict)

def test_MentalHealthClassifier_predict(newsgroups_data):
    """

    """
    ## Parse Fixture
    test_files, label_dict = newsgroups_data
    ## Fit Model
    model = MentalHealthClassifier("depression",
                                   "logistic",
                                   model_kwargs={"random_state":42},
                                   vocab_kwargs={"max_vocab_size":10})
    model, y_train_pred = model.fit(test_files, label_dict, True)
    ## Make Predictions
    y_test_pred = model.predict(test_files)
    ## Tests
    assert isinstance(y_test_pred, dict)
    assert y_train_pred == y_test_pred
    y_true_vector = model._vectorize_labels(test_files, label_dict, "depression")
    y_pred_vector = np.array([y_train_pred[f] > 0.5 for f in test_files]).astype(int)
    accuracy = accuracy_score(y_true_vector, y_pred_vector)
    assert accuracy > 0.5
    ## Fit Classifier Model
    model = MentalHealthClassifier("depression",
                                 "svm",
                                 model_kwargs={"random_state":42},
                                 vocab_kwargs={"max_vocab_size":10})
    model = model.fit(test_files, label_dict, False)
    y_train_pred = model.predict(test_files)
    svm_y_pred_vector = np.array([y_train_pred[f] > 0.5 for f in test_files]).astype(int)
    svm_accuracy = accuracy_score(y_true_vector, svm_y_pred_vector)
    assert svm_accuracy > 0.5

def test_MentalHealthClassifier_copy():
    """

    """
    ## Initialize Models
    model = MentalHealthClassifier("depression")
    different_model = model.copy()
    ## Test
    assert hash(model) != hash(different_model)
    assert model.__repr__() == different_model.__repr__()

def test_MentalHealthClassifier_dump(newsgroups_data):
    """

    """
    ## Fit Model
    test_files, label_dict = newsgroups_data
    model = MentalHealthClassifier("depression", vocab_kwargs={"max_vocab_size":10})
    model, training_preds = model.fit(test_files, label_dict, True)
    ## Dump Model
    base_path = os.path.dirname(__file__) + "/../data/"
    dump_path = f"{base_path}temp_model"
    model.dump(dump_path)
    ## Load Model
    model_loaded = joblib.load(dump_path + ".joblib")
    loaded_training_preds = model_loaded.predict(test_files)
    ## Test
    assert training_preds == loaded_training_preds
    ## Remove Test File
    _ = os.system(f"rm {dump_path}.joblib")