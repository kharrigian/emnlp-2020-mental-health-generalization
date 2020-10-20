
########################
### Imports
########################

## Standard Library
import os
import gzip
import json
from copy import deepcopy
from datetime import datetime

## External Libraries
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

## Local Modules
from mhlib.model.vocab import Vocabulary
from mhlib.preprocess.tokenizer import Tokenizer
from mhlib.model.file_vectorizer import File2Vec
from mhlib.model.feature_selectors import KLDivergenceSelector, FeatureSelector

########################
### Fixtures
########################

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
                                          "sci.electronics",
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
                          keep_retweets=False)
    processing_time = int(datetime.utcnow().timestamp())
    label_dict = {}
    for t, (text, target) in enumerate(zip(data["data"], data["target"])):
        target_name = data["target_names"][target]
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
    ## Create Vocabulary Object and Fit
    vocab = Vocabulary()
    vocab = vocab.fit(data_files, chunksize=10, jobs=8)
    ## Vectorize Data
    f2v = File2Vec(favor_dense=False)
    f2v.vocab = vocab
    f2v._initialize_dict_vectorizer()
    filenames, X = f2v._vectorize_files(data_files)
    class_map = dict((n, i) for i,n in enumerate(data["target_names"]))
    y = np.array([class_map[label_dict[f]] for f in filenames])
    yield X, y, vocab, class_map, processed_data_folder
    ## Teardown
    _ = os.system(f"rm -rf {processed_data_folder}")

########################
### Testing
########################

def test_kldivergence_selector(newsgroups_data):
    """

    """
    ## Get Data Components
    X, y, vocab, class_map, _ = newsgroups_data
    ## Copy the Vocab Object as not Overwrie
    vocab = deepcopy(vocab)
    ## Initialize Selector
    selector = KLDivergenceSelector(vocab=vocab,
                                    top_k=250,
                                    min_support=3,
                                    stopwords=None,
                                    add_lambda=0.01,
                                    beta=0)
    ## Fit and Transform
    X_T = selector.fit_transform(X, y)
    ## Check
    assert len(selector.vocab.vocab) == 250
    div_df = pd.DataFrame(selector._div_score, columns = selector._vocabulary_terms).T
    div_df.rename(columns=dict((y,x) for x,y in class_map.items()), inplace=True)
    assert div_df.sum(axis=1).nsmallest(10).index.tolist() == ['our',
                'then',
                'though',
                'much',
                'more',
                'lot',
                'as',
                'not_have',
                'follow',
                'that']
    assert ['hockey',
            'circuit',
            'nhl',
            'playoffs',
            'rangers',
            'playoff',
            'leafs',
            'amp',
            'wings',
            'toronto']
    ## Transform Test
    assert X_T.shape == (X.shape[0], 250)
    ## Repr Test
    assert selector.__repr__() == 'KLDivergenceSelector(top_k=250, min_support=3, add_lambda=0.01, beta=0)'

def test_general_feature_selector(newsgroups_data):
    """

    """
    ## Get Data Components
    X, y, vocab, class_map, _ = newsgroups_data
    ## Copy the Vocab Object as not Overwrie
    vocab = deepcopy(vocab)
    ## Initialize Selector
    selector = FeatureSelector(vocab = vocab,
                               selector = "kldivergence",
                               feature_selection_kwargs = dict(top_k=300,
                                                               min_support=3,
                                                               stopwords=None,
                                                               add_lambda=0.01,
                                                               beta=0))
    ## Repr Test
    assert selector.__repr__() == 'FeatureSelector(selector=KLDivergenceSelector(top_k=300, min_support=3, add_lambda=0.01, beta=0))'
    ## Fit Transform Test
    X_T = selector.fit_transform(X, y)
    assert X_T.shape == (X.shape[0], 300)
    ## No Selection Test
    no_selector = FeatureSelector(vocab = vocab,
                                  selector = None,
                                  feature_selection_kwargs = {})
    X_T = no_selector.fit_transform(X, y)
    assert X_T.shape == X.shape
