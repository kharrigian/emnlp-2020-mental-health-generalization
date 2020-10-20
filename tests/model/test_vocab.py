
#####################
### Imports
#####################

## Standard Library
import json
import os
import gzip
from collections import Counter

## External Libraries
import pytest

## Local
from mhlib.model.vocab import Vocabulary
from mhlib.util.helpers import flatten

######################
### Globals
######################

DEFAULT_VOCAB_INIT = dict(
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
                 max_vocab_size=100000,
                 min_token_freq=0,
                 max_token_freq=None,
                 ngrams=(1,1),
                 max_tokens_per_document=None,
                 max_documents_per_user=None)

######################
### Fixtures
######################

@pytest.fixture(scope="module")
def test_files():
    """

    """
    ## Create Fake Data (2 Users)
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
    {'user_id_str': 'FAKE_USER_1',
     'created_utc': 14143309301,
     'text': 'Ready to give up :( #Depression',
     'text_tokenized': ['ready', 'to', 'give', 'up', ':(', 'depression'],
     'tweet_id': 'NOT_A_REAL_TWEET_ID_2',
     'depression': 'depression',
     'gender': 'F',
     'age': 31.20056089,
     'entity_type': 'tweet',
     'date_processed_utc': 1576219763,
     'source': 'FAKE_SOURCE_PATH.tweets.gz',
     'dataset': ['FAKE_DATASET']},     
    {'user_id_str': 'FAKE_USER_1',
     'created_utc': 14143309501,
     'text': 'NOT GOING TO SCHOOL TODAY. OR HOPEFULLY EVER AGAIN! PEACE 2019.',
     'text_tokenized': ['not_going','to','school','today','.', 'or', 'hopefully', 'ever', 'again', '!', 'peace', '<NUMERIC>', '.', '<UPPER_FLAG>', '<NEGATE_FLAG>'],
     'tweet_id': 'NOT_A_REAL_TWEET_ID_3',
     'depression': 'depression',
     'gender': 'F',
     'age': 31.20056089,
     'entity_type': 'tweet',
     'date_processed_utc': 1576219763,
     'source': 'FAKE_SOURCE_PATH.tweets.gz',
     'dataset': ['FAKE_DATASET']
    },
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
    },    
    {'user_id_str': 'FAKE_USER_2',
     'created_utc': 14143309501,
     'text': 'Nevermind. Avengers was quite the let down. Wouldn\'t you agree @FAKE_FRIEND2?',
     'text_tokenized': ['nevermind','.','avengers','was','quite','the','let','down','.','would', 'not_you', 'agree','<USER_MENTION>','?','<NEGATE_FLAG>'],
     'tweet_id': 'NOT_A_REAL_TWEET_ID_5',
     'depression': 'control',
     'gender': 'M',
     'age': 25.24443,
     'entity_type': 'tweet',
     'date_processed_utc': 1576219763,
     'source': 'FAKE_SOURCE_PATH.tweets.gz',
     'dataset': ['FAKE_DATASET']
    },            
    ]
    ## Create Fake Data Files
    base_path = os.path.dirname(__file__) + "/../data/"
    filenames = []
    for user_id_str in ["FAKE_USER_1","FAKE_USER_2"]:
        user_data = [f for f in fake_data if f["user_id_str"] == user_id_str]
        user_file = f"{base_path}{user_id_str}.tweets.json.tar.gz"
        filenames.append(user_file)
        with gzip.open(user_file, "wt", encoding="utf-8") as the_file:
            json.dump(user_data, the_file)
    yield filenames
    ## Teardown
    for f in filenames:
        _ = os.system(f"rm {f}")


@pytest.fixture(scope="module")
def raw_token_counts():
    """

    """
    counts = {
        "FAKE_USER_1":Counter({
                              ('<NEGATE_FLAG>',): 2, 
                              ('to',): 2, 
                              ('.',): 2, 
                              ('<RETWEET>',): 1, 
                              (':',): 1, 
                              ('i',): 1, 
                              ('can',): 1, 
                              ('not_believe',): 1, 
                              ('<USER_MENTION>',): 1, 
                              ('said',): 1, 
                              ('this',): 1, 
                              ('💔',): 1, 
                              ('<URL_TOKEN>',): 1, 
                              ('ready',): 1, 
                              ('give',): 1, 
                              ('up',): 1, 
                              (':(',): 1, 
                              ('depression',): 1, 
                              ('not_going',): 1, 
                              ('school',): 1, 
                              ('today',): 1, 
                              ('or',): 1, 
                              ('hopefully',): 1, 
                              ('ever',): 1, 
                              ('again',): 1, 
                              ('!',): 1, 
                              ('peace',): 1, 
                              ('<NUMERIC>',): 1, 
                              ('<UPPER_FLAG>',): 1}),
        "FAKE_USER_2":Counter({
                              ('the',): 2, 
                              ('avengers',): 2, 
                              ('.',): 2, 
                              ('excited',): 1, 
                              ('to',): 1, 
                              ('see',): 1, 
                              ('new',): 1, 
                              ('movie',): 1, 
                              ('tonight',): 1, 
                              ('!',): 1, 
                              ('nevermind',): 1, 
                              ('was',): 1, 
                              ('quite',): 1, 
                              ('let',): 1, 
                              ('down',): 1, 
                              ('would',): 1, 
                              ('not_you',): 1, 
                              ('agree',): 1, 
                              ('<USER_MENTION>',): 1, 
                              ('?',): 1, 
                              ('<NEGATE_FLAG>',): 1}),
    }
    return counts
        
######################
### Tests
######################

def test_load_and_count(test_files,
                        raw_token_counts):
    """

    """
    ## Get Filenames
    filenames = test_files
    ## Get Full Unigram Token Counts
    token_counts = raw_token_counts
    ## Initialize Vocabulary Object
    vocab = Vocabulary(**DEFAULT_VOCAB_INIT)
    ## Load Counts for each User
    vocab_counts = [vocab._load_and_count(f) for f in filenames]
    ## Check
    assert vocab_counts[0] == token_counts["FAKE_USER_1"]
    assert vocab_counts[1] == token_counts["FAKE_USER_2"]


def test_get_ngrams():
    """

    """
    ## Initialize Vocabulary Object
    vocab = Vocabulary(**DEFAULT_VOCAB_INIT)
    ## Test Raw Tokens
    raw_tokens = ["A","<NUMERIC>","C","D","<UPPER_FLAG>","<NEGATE_FLAG>"]
    ## Tests
    assert vocab.get_ngrams(raw_tokens, 1, 1) == [('A',), ('<NUMERIC>',), ('C',), ('D',), ('<UPPER_FLAG>',), ('<NEGATE_FLAG>',)]
    assert vocab.get_ngrams(raw_tokens, 1, 2) == [('A',), ('<NUMERIC>',), ('C',), ('D',), ('A', '<NUMERIC>'), ('<NUMERIC>', 'C'), ('C', 'D'), ('<UPPER_FLAG>',), ('<NEGATE_FLAG>',)]
    assert vocab.get_ngrams(raw_tokens, 2, 3) == [('A', '<NUMERIC>'), ('<NUMERIC>', 'C'), ('C', 'D'), ('A', '<NUMERIC>', 'C'), ('<NUMERIC>', 'C', 'D'), ('<UPPER_FLAG>',), ('<NEGATE_FLAG>',)]
    ## Test Param Misspecficiation
    with pytest.raises(ValueError):
        _ = vocab.get_ngrams(raw_tokens, 2, 1)
    with pytest.raises(ValueError):
        _ = vocab.get_ngrams(raw_tokens, 0, 1)


def test_fit(test_files,
             raw_token_counts):
    """

    """
    ## Initialize Vocabulary Object
    init_params = DEFAULT_VOCAB_INIT.copy()
    init_params["max_vocab_size"] = None
    vocab = Vocabulary(**init_params)
    ## Learn Vocabulary
    vocab = vocab.fit(test_files, chunksize=1, jobs=1)
    ## Test
    expected_terms = set(flatten([i.keys() for i in raw_token_counts.values()]))
    assert vocab.vocab == expected_terms
    assert dict((y, x) for x,y in enumerate(sorted(expected_terms))) == vocab.ngram_to_idx

def test_max_vocab_size(test_files,
                        raw_token_counts):
    """

    """
    ## Initialize Vocabulary Object
    init_params = DEFAULT_VOCAB_INIT.copy()
    init_params["max_vocab_size"] = 1
    vocab = Vocabulary(**init_params)
    ## Learn Vocabulary
    vocab = vocab.fit(test_files, chunksize=1, jobs=1)
    ## Count Expected Vocab
    c_gt = Counter()
    for t in raw_token_counts.values():
        c_gt += t
    assert set(c_gt.most_common(1)[0][:1]) == vocab.vocab # ('.', )

def test_min_token_freq(test_files):
    """

    """
    ## Initialize Vocabulary Object
    init_params = DEFAULT_VOCAB_INIT.copy()
    init_params["min_token_freq"] = 3
    vocab = Vocabulary(**init_params)
    ## Learn Vocabulary
    vocab = vocab.fit(test_files, chunksize=1, jobs=1)
    ## Test
    assert vocab.vocab == {('.',), ('<NEGATE_FLAG>',), ('to',)}
    
def test_max_token_freq(test_files):
    """

    """
    ## Initialize Vocabulary Object
    init_params = DEFAULT_VOCAB_INIT.copy()
    init_params["max_token_freq"] = 0
    vocab = Vocabulary(**init_params)
    ## Learn Vocabulary
    vocab = vocab.fit(test_files, chunksize=1, jobs=1)
    ## Test
    assert vocab.vocab == set()

def test_max_tokens_per_document(test_files):
    """

    """
    ## Initialize Vocabulary Object
    init_params = DEFAULT_VOCAB_INIT.copy()
    init_params["max_tokens_per_document"] = 1
    vocab = Vocabulary(**init_params)
    ## Learn Vocabulary
    vocab = vocab.fit(test_files, chunksize=1, jobs=1)
    ## Test
    assert vocab.vocab == {('<RETWEET>',), ('ready',), ('not_going',), ('excited',), ('nevermind',)} # First tokens of processed data

def test_max_documents_per_user(test_files):
    """

    """
    ## Initialize Vocabulary Object
    init_params = DEFAULT_VOCAB_INIT.copy()
    init_params["max_documents_per_user"] = 1
    vocab = Vocabulary(**init_params)
    ## Learn Vocabulary
    vocab = vocab.fit(test_files, chunksize=1, jobs=1)
    ## Test
    expected_missing = ["depression","agree","ready","not_believe"] # Not in the most recent post
    for e in expected_missing:
        assert (e, ) not in vocab.vocab

def test_initialize_with_ngrams(test_files):
    """

    """
    ## Initialize Vocabulary Object
    init_params = DEFAULT_VOCAB_INIT.copy()
    init_params["ngrams"] = (2, 2)
    init_params["max_documents_per_user"] = 1
    init_params["max_tokens_per_document"] = 3
    vocab = Vocabulary(**init_params)
    ## Learn Vocabulary
    vocab = vocab.fit(test_files[:1])
    ## Check
    assert vocab.vocab == {('not_going', 'to'), ('to', 'school')}

def test_repr():
    """

    """
    ## Initialize
    vocab = Vocabulary(**DEFAULT_VOCAB_INIT)
    ## Get repr string
    repr_str = vocab.__repr__()
    ## Check
    assert repr_str.startswith("Vocabulary(")

def test_get_ordered_vocabulary(test_files):
    """

    """
    ## Initialize Vocabulary Object
    init_params = DEFAULT_VOCAB_INIT.copy()
    init_params["ngrams"] = (1, 1)
    init_params["max_documents_per_user"] = 1
    init_params["max_tokens_per_document"] = 3
    vocab = Vocabulary(**init_params)
    ## Learn Vocabulary
    vocab = vocab.fit(test_files[:1])
    ## Check
    ordered_vocab = vocab.get_ordered_vocabulary()
    assert ordered_vocab == [('not_going',), ('school',), ('to',)] == sorted(vocab.vocab)
