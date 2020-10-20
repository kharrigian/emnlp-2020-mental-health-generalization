
#########################
### Imports
#########################

## Standard Libraries
import os
import json
import gzip

## External Libraries
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

## Local Modules
from mhlib.model.vocab import Vocabulary
from mhlib.model.file_vectorizer import File2Vec

#########################
### Globals
#########################

## Vocabulary Initialization Parameters
DEFAULT_VOCAB_PARAMS = dict(
                 filter_negate=False,
                 filter_upper=False,
                 filter_punctuation=False,
                 filter_numeric=False,
                 filter_user_mentions=False,
                 filter_url=False,
                 filter_retweet=False,
                 filter_stopwords=False,
                 keep_pronouns=True,
                 preserve_case=True,
                 max_vocab_size=None,
                 min_token_freq=0,
                 max_token_freq=None,
                 ngrams=(1,1),
                 max_tokens_per_document=None,
                 max_documents_per_user=None)


#########################
### Fixtures
#########################

@pytest.fixture(scope="module")
def sample_data():
    """

    """
    ## Cache Path
    base_path = os.path.dirname(__file__) + "/../data/"
    ## Fake Data
    fake_data = [
    {'user_id_str': 'FAKE_USER_1',
     'created_utc': 14142309301,
     'text': 'RT: I can\'t believe @FAKE_MENTION said this 💔 www.google.com',
     'text_tokenized':['<RETWEET>',':', 'i', 'can','not_believe', '<USER_MENTION>', 'said', 'this', '💔', '<URL_TOKEN>','<NEGATE_FLAG>'],
     'tweet_id': 'NOT_A_REAL_TWEET_ID_1',
     'depression': 'depression',
     'gender': 'F',
     'age': 31.20056089,
     'entity_type': 'tweet',
     'date_processed_utc': 1576219763,
     'source': 'FAKE_SOURCE_PATH.tweets.gz',
     'dataset': ['FAKE_DATASET']},
    {'user_id_str': 'FAKE_USER_2',
     'created_utc': 14143309501,
     'text': 'Excited to see the new Avengers movie tonight!',
     'text_tokenized': ['excited', 'to', 'see', 'the', 'new', 'avengers', 'movie', 'tonight', '!'],
     'tweet_id': 'NOT_A_REAL_TWEET_ID_4',
     'depression': 'control',
     'gender': 'M',
     'age': 25.24443,
     'entity_type': 'tweet',
     'date_processed_utc': 1576219743,
     'source': 'FAKE_SOURCE_PATH.tweets.gz',
     'dataset': ['FAKE_DATASET']
    }]
    ## Dump Fake Data
    filenames = []
    for row in fake_data:
        user = row["user_id_str"]
        row_file = f"{base_path}{user}.tweets.jzon.gz"
        filenames.append(row_file)
        with gzip.open(row_file, "wt", encoding="utf-8") as the_file:
            json.dump([row], the_file)
    yield filenames, fake_data
    ## Teardown
    for f in filenames:
        _ = os.system(f"rm {f}")

@pytest.fixture(scope="module")
def embeddings_transformer(sample_data):
    """

    """
    ## Parse Fixture
    fake_files, fake_data = sample_data
    ## Get Embeddings Object
    embeddings = _initialize_embedding(fake_files, 25, "mean")
    return embeddings, fake_files, fake_data

#########################
### Helpers
#########################

def _initialize_f2v(fake_files):
    """

    """
    ## Initialize
    f2v = File2Vec(DEFAULT_VOCAB_PARAMS, True)
    ## Learn Vocabulary
    vocab = Vocabulary(DEFAULT_VOCAB_PARAMS)
    vocab.fit(fake_files)
    ## Assign Vocabulary to F2V
    f2v.vocab = vocab
    ## Initialize Dict Vectorizer
    f2v._initialize_dict_vectorizer()
    return f2v

#########################
### Tests
#########################

def test_File2Vec_repr():
    """

    """
    ## Initialize
    f2v = File2Vec(vocab_kwargs=DEFAULT_VOCAB_PARAMS,
                   favor_dense=True)
    ## Test repr
    f2v_repr = f2v.__repr__()
    assert f2v_repr == f"File2Vec(vocab_kwargs={DEFAULT_VOCAB_PARAMS}, favor_dense=True)"

def test_File2Vec_initialize_dict_vectorizer(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, fake_data = sample_data
    ## Initialize F2v
    f2v = _initialize_f2v(fake_files)
    ## Test Dict Vectorizer
    assert hasattr(f2v, "_count2vec")
    dvec = f2v._count2vec
    assert isinstance(dvec, DictVectorizer)
    assert hasattr(dvec, "vocabulary_")
    assert dvec.vocabulary_ == f2v.vocab.ngram_to_idx
    assert hasattr(dvec, "feature_names_")
    outvec = dvec.transform([{("<USER_MENTION>",):True}])
    assert outvec.toarray()[0, f2v.vocab.ngram_to_idx[("<USER_MENTION>",)]] == 1
    assert outvec.sum() == 1


def test_File2Vec_vectorize_single_file(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, fake_data = sample_data
    ## Initialize F2V
    f2v = _initialize_f2v(fake_files)
    ## Vectorize File
    test_file = fake_files[0]
    outfile, outvector = f2v._vectorize_single_file(test_file)
    ## Tests
    assert test_file == outfile
    assert outvector.shape[1] == len(f2v.vocab.vocab)
    assert isinstance(outvector, csr_matrix)
    outvector = outvector.toarray()[0]
    for ngram, idx in f2v.vocab.ngram_to_idx.items():
        if ngram[0] in fake_data[0]["text_tokenized"]:
            assert outvector[idx] == 1
        else:
            assert outvector[idx] == 0

def test_File2Vec_vectorize_files(sample_data):
    """

    """
    # Separate Fixture
    fake_files, fake_data = sample_data
    ## Initialize F2V
    f2v = _initialize_f2v(fake_files)
    ## Vectorize Files
    outfiles, outvecs = f2v._vectorize_files(fake_files)
    assert all(f in outfiles for f in fake_files)
    assert outvecs.shape == (2, len(f2v.vocab.vocab))
    for f, ov in zip(outfiles, outvecs):
        f_ind = [i for i, x in enumerate(fake_files) if x == f][0]
        fake_data_tokens = fake_data[f_ind]["text_tokenized"]
        for ngram, idx in f2v.vocab.ngram_to_idx.items():
            if ngram[0] in fake_data_tokens:
                assert ov[idx] == 1
            else:
                assert ov[idx] == 0

def test_File2Vec_vectorize_labels():
    """

    """
    ## Initialize F2V
    f2v = File2Vec()
    ## Vectorize Labels
    label_dict = {"file_1":"depression",
                  "file_2":"control",
                  "file_3":"depression"}
    key_order = ["file_3","file_1","file_2"]
    labels = f2v._vectorize_labels(key_order, label_dict, "depression")
    ## Test
    assert np.all(labels == np.array([1, 1, 0]))
