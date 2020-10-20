
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
from mhlib.model.feature_extractors import (DummyTransformer,
                                            LIWCTransformer,
                                            EmbeddingTransformer,
                                            FeaturePreprocessor)
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

def _intialize_liwc(fake_files,
                    norm="max"):
    """

    """
    ## Learn Vocabulary
    vocab = Vocabulary(DEFAULT_VOCAB_PARAMS)
    vocab.fit(fake_files)
    ## Initialize LIWC
    liwc = LIWCTransformer(vocab=vocab,
                           norm=norm)
    return liwc

def _initialize_embedding(fake_files,
                          dim = 25,
                          pooling = "mean"):
    """

    """
    ## Learn Vocabulary
    vocab = Vocabulary(DEFAULT_VOCAB_PARAMS)
    vocab.fit(fake_files)
    ## Initialize Embeddding Transformer
    embeddings = EmbeddingTransformer(vocab,
                                      dim,
                                      pooling)
    return embeddings

#########################
### Tests
#########################

def test_DummyTransformer_repr():
    """

    """
    ## Initialize Transformer
    dt = DummyTransformer()
    ## Test
    dt_repr = dt.__repr__()
    assert dt_repr == "DummyTransformer()"

def test_DummyTransformer_fit():
    """

    """
    ## Initialize Transformer
    dt = DummyTransformer()
    ## Fit
    X = np.array([[1, 1, 0]])
    res = dt.fit(X)
    assert res == dt

def test_DummyTransformer_transform():
    """

    """
    ## Initialize Transformer
    dt = DummyTransformer()
    ## Fit
    X = np.array([[1, 1, 0]])
    dt.fit(X)
    X_T = dt.transform(X)
    assert np.all(X == X_T)

def test_DummyTransformer_fit_transform():
    """

    """
    ## Initialize Transformer
    dt = DummyTransformer()
    ## Fit Transform
    X = np.array([[1, 1, 0]])
    X_T = dt.fit_transform(X)
    ## Test
    assert np.all(X == X_T)

def test_LIWCTransformer_repr(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, _ = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    ## Test REPR
    liwc_repr = liwc.__repr__()
    assert liwc_repr == f"LIWCTransformer(vocab={liwc.vocab}, norm=max)"

def test_LIWCTransformer_initialize_dictionary(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, _ = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    ## Tests
    assert len(liwc.cache) == 2144
    assert len(liwc.dimensions) == 64
    assert isinstance(liwc.dimensions, dict)
    assert len(liwc.names) == 64
    assert isinstance(liwc.names, list)
    assert isinstance(liwc.names[0], str)

def test_LIWCTransformer_matches(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, _ = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    ## Tests
    assert liwc.matches("Hellos","Hell")
    assert not liwc.matches("Hellos","hell")   
    assert not liwc.matches("Hellos", "ellos")

def test_LIWCTransformer_add(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, _ = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    ## Dimension Counts
    counts = np.zeros(64)
    ## Add To Counts
    for d in liwc.dimensions.keys():
        liwc.add(counts, [d])
    ## Test
    assert np.all(counts) == 1

def test_LIWCTransformer_search(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, _ = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    expected_entry = ('anorexi*', ['146', '148', '150']) # (Leading Term, Dimensions)
    ## Search
    liwc_res = liwc.search("anorexic")
    ## Test
    assert liwc_res == expected_entry[1]

def test_LIWCTransformer_binary_search(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, _ = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    expected_entry = ('anorexi*', ['146', '148', '150']) # (Leading Term, Dimensions)
    ## Search 
    assert liwc.binary("anorex") is None
    assert liwc.binary("anorexi") == expected_entry[1]

def test_LIWCTransformer_classify(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, _ = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    ## Run Classify
    cls_res = liwc.classify(["anorexic", "FAKE_WORD"])
    ## Tests
    expected_dims = set(['146', '148', '150'])
    assert isinstance(cls_res, tuple)
    assert cls_res[0] == 2
    assert cls_res[1] == 1
    assert sum(cls_res[2]) == 3
    for dim, ind in liwc.dimensions.items():
        if dim in expected_dims:
            assert cls_res[2][ind] == 1
        else:
            assert cls_res[2][ind] == 0

def test_LIWCTransformer_fit(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, fake_data = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    ## Get X
    f2v = _initialize_f2v(fake_files)
    outfiles, X = f2v._vectorize_files(fake_files)
    ## Fit
    liwc = liwc.fit(X)
    ## Tests
    assert isinstance(liwc._dim_map, np.ndarray)
    assert liwc._dim_map.shape == (len(f2v.vocab.vocab), 64)
    assert np.all(liwc.classify([i[0].lower() for i in liwc.vocab.get_ordered_vocabulary()])[2] == \
                  liwc._dim_map.sum(axis=0))

def test_LIWCTransformer_transform(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, fake_data = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    ## Get X
    f2v = _initialize_f2v(fake_files)
    outfiles, X = f2v._vectorize_files(fake_files)
    ## Fit
    liwc = liwc.fit(X)
    ## Apply Transformation
    X_T = liwc.transform(X)
    ## Tests
    assert isinstance(X_T, np.ndarray)
    assert X_T.shape == (2, 64)
    assert np.all(X_T.max(axis=1) == 1)
    for k, f in enumerate(fake_files):
        f_ind = [i for i, x in enumerate(outfiles) if x == f][0]
        input_tokens = fake_data[k]["text_tokenized"]
        f_x = X_T[f_ind]
        liwc_t = liwc.classify(input_tokens)[2]
        assert np.all((liwc_t / liwc_t.max()) == f_x)

def test_LIWCTransform_fit_transform(sample_data):
    """

    """
    ## Separate Fixture
    fake_files, fake_data = sample_data
    ## Initialize
    liwc = _intialize_liwc(fake_files, "max")
    ## Get X
    f2v = _initialize_f2v(fake_files)
    outfiles, X = f2v._vectorize_files(fake_files)
    ## Fit and Apply Transformation
    X_T = liwc.fit_transform(X)
    ## Tests
    assert isinstance(X_T, np.ndarray)
    assert X_T.shape == (2, 64)
    assert np.all(X_T.max(axis=1) == 1)
    for k, f in enumerate(fake_files):
        f_ind = [i for i, x in enumerate(outfiles) if x == f][0]
        input_tokens = fake_data[k]["text_tokenized"]
        f_x = X_T[f_ind]
        liwc_t = liwc.classify(input_tokens)[2]
        assert np.all((liwc_t / liwc_t.max()) == f_x)

def test_EmbeddingTransformer_repr(embeddings_transformer):
    """

    """
    ## Split Fixture
    embeddings, _, _ = embeddings_transformer
    ## Get REPR
    em_repr = embeddings.__repr__()
    assert em_repr == f"EmbeddingTransform(vocab={embeddings.vocab}, pooling=mean, dim=25)"

def test_EmbeddingTransformer_misparameterization():
    """

    """
    with pytest.raises(ValueError):
        _ = EmbeddingTransformer(None, 25, "mode")
    with pytest.raises(ValueError):
        _ = EmbeddingTransformer(None, 300)
    

def test_EmbeddingTransformer_load_embeddings(embeddings_transformer):
    """"

    """
    ## Split Fixture
    embeddings, _, _ = embeddings_transformer
    ## Tests
    assert isinstance(embeddings.embedding_matrix, np.ndarray)
    assert embeddings.embedding_matrix.shape == (len(embeddings.vocab.vocab), 25)
    assert len(embeddings._matched_tokens) == sum(np.all(embeddings.embedding_matrix!=0,axis=1))

def test_EmbeddingTransformer_abs_maxND(embeddings_transformer):
    """

    """
    ## Split Fixture
    embeddings, _, _ = embeddings_transformer
    ## Data Objects for Text
    x_test = np.array([[-2, 1, -1, 2],
                       [-3, 0, 2, -3]])
    x_abs_max0 = embeddings._absmaxND(x_test, axis=0)
    x_abs_max1 = embeddings._absmaxND(x_test, axis=1)
    ## Tests
    assert np.all(x_abs_max0 == np.array([-3, 1, 2, -3]))
    assert np.all(x_abs_max1 == np.array([2, -3]))

def test_EmbeddingTransformer_fit(embeddings_transformer):
    """

    """
    ## Split Fixture
    embeddings, _, _ = embeddings_transformer
    ## Fit (Just Returns Self)
    ret = embeddings.fit(None)
    assert ret == embeddings

def test_EmbeddingTransformer_fit_transform(embeddings_transformer):
    """

    """
    ## Split Fixture
    embeddings, fake_files, fake_data = embeddings_transformer
    ## Create X
    f2v = _initialize_f2v(fake_files)
    outfiles, X = f2v._vectorize_files(fake_files)
    if fake_files != outfiles:
        fake_files = fake_files[::-1]
        fake_data = fake_data[::-1]
    X = csr_matrix(X)
    ## Transform X (Mean Pooling)
    embeddings.pooling = "mean"
    X_T_mean = embeddings.fit_transform(X)
    ## Transform X (Max Pooling)
    embeddings.pooling = "max"
    X_T_max = embeddings.fit_transform(X)
    ## Check
    for x, xt_mean, xt_max in zip(X, X_T_mean, X_T_max):
        xvec = x.toarray()[0]
        matched_tokens = [embeddings.vocab.ngram_to_idx[i] for i in embeddings._matched_tokens]
        matched_token_count = x.toarray()[:, matched_tokens].sum(axis=1, keepdims=True)
        assert np.all(np.isclose(embeddings.embedding_matrix[np.nonzero(xvec)].sum(axis=0) / matched_token_count, xt_mean))
        assert np.all(np.isclose(embeddings._absmaxND(embeddings.embedding_matrix[np.nonzero(xvec)], axis=0), xt_max))

def test_FeaturePreprocessor_repr():
    """

    """
    ## Initialize Preprocessor
    preprocessor = FeaturePreprocessor(None,
                                       feature_flags={},
                                       feature_kwargs={},
                                       standardize=True)
    ## Test
    pre_repr = preprocessor.__repr__()
    assert pre_repr == "FeaturePreprocessor(vocab=None, feature_flags={}, feature_kwargs={})"

def test_FeaturePreprocessor_validate_preprocessing_params():
    """

    """
    ## Non-integer Glove Dimension
    with pytest.raises(ValueError):
        _ = FeaturePreprocessor(None, {"glove":True}, {"glove":{"dim":None}})
    ## Not Supported Glove Dimensions
    with pytest.raises(ValueError):
        _ = FeaturePreprocessor(None, {"glove":True}, {"glove":{"dim":300}})
    ## Non-string Norm
    with pytest.raises(ValueError):
        _ = FeaturePreprocessor(None, {"liwc":True}, {"liwc":{"norm":20}})

def test_FeaturePreprocessor_initialize_transformer(sample_data):
    """

    """
    ## Parse Sample Data Fixture
    fake_files, fake_data = sample_data
    ## Initialize Vocab
    vocab = Vocabulary(**DEFAULT_VOCAB_PARAMS)
    vocab.fit(fake_files)
    ## Initialize Feature Preprocessor
    fp = FeaturePreprocessor(vocab,
                             {"tfidf":True,"liwc":True},
                             standardize=False)
    ## Test
    assert isinstance(fp._transformers, dict)
    assert "tfidf" in fp._transformers
    assert "liwc" in fp._transformers
    assert "standardize" not in fp._transformers

def test_FeaturePreprocessor_initialize_preprocessing_pipeline(sample_data):
    """

    """
    ## Parse Sample Data Fixture
    fake_files, fake_data = sample_data
    ## Initialize Vocab
    vocab = Vocabulary(**DEFAULT_VOCAB_PARAMS)
    vocab.fit(fake_files)
    ## Initialize Feature Preprocessor
    fp = FeaturePreprocessor(vocab, {}, {}, standardize=True)
    ## Check
    assert isinstance(fp._transformers, dict)
    assert "standardize" in fp._transformers
    assert isinstance(fp._transformers.get("standardize"), StandardScaler)
    assert isinstance(fp._transformer_names, list)
    assert len(fp._transformer_names) == 0

def test_FeaturePreprocessor_fit(sample_data):
    """

    """
    ## Parse Sample Data Fixture
    fake_files, fake_data = sample_data
    ## Get Document-Term Matrix
    f2v = _initialize_f2v(fake_files)
    outfiles, X = f2v._vectorize_files(fake_files)
    ## Initialize Feature Preprocessor
    fp = FeaturePreprocessor(f2v.vocab,
                             {"tfidf":True,
                              "liwc":True,
                              "glove":False,
                              "bag_of_words":True,
                              "lda":True},
                              {},
                              standardize=True)
    ## Fit Model
    fp = fp.fit(X)
    ## Tests
    assert fp._transformers["tfidf"].__getstate__()["_idf_diag"].shape == (len(f2v.vocab.vocab), len(f2v.vocab.vocab))
    assert isinstance(fp._transformers["liwc"]._dim_map, np.ndarray)
    assert "glove" not in fp._transformers
    assert isinstance(fp._transformers.get("bag_of_words"), DummyTransformer)
    assert isinstance(fp._transformers["lda"].components_, np.ndarray)
    assert isinstance(fp._transformers["standardize"].mean_, np.ndarray)

def test_FeaturePreprocessor_fit_transform(sample_data):
    """

    """
    ## Parse Sample Data Fixture
    fake_files, fake_data = sample_data
    ## Get Document-Term Matrix
    f2v = _initialize_f2v(fake_files)
    outfiles, X = f2v._vectorize_files(fake_files)
    ## Initialize Feature Preprocessor
    fp = FeaturePreprocessor(f2v.vocab,
                             {"tfidf":True,
                              "liwc":True,
                              "glove":False,
                              "bag_of_words":True,
                              "lda":True},
                              {},
                              standardize=True)
    ## Fit Model and Transform Data
    X = csr_matrix(X)
    X_T = fp.fit_transform(X)
    ## Tests
    assert isinstance(X_T, np.ndarray)
    assert X_T.shape[0] == X.shape[0]
    assert X_T.shape[1] == X.shape[1] * 2 + fp._transformers["lda"].n_components + 64
    assert np.all(np.isclose(X_T.mean(axis=0), 0))
