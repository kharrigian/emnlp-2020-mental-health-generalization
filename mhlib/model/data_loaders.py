
"""
Data loaders for outputs generated from the preprocessing pipeline.
"""

#################
### Imports
#################

## Standard Libary
import os
import json
import gzip
from datetime import datetime
from string import punctuation

## External Libraries
import emoji
import pandas as pd
import numpy as np

## Local Modules
from ..preprocess.tokenizer import (STOPWORDS,
                                    PRONOUNS,
                                    CONTRACTIONS)
from ..util.helpers import flatten

#################
### Globals
#################

## Mental Health Subreddit/Term Filters
RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../data/resources/"
MH_SUBREDDIT_FILE = f"{RESOURCE_DIR}mh_subreddits.json"
MH_TERMS_FILE = f"{RESOURCE_DIR}mh_terms.json"

## Load Mental Health Subreddits
with open(MH_SUBREDDIT_FILE, "r") as the_file:
    MH_SUBREDDITS = json.load(the_file)

## Load Mental Health Terms
with open(MH_TERMS_FILE, "r") as the_file:
    MH_TERMS = json.load(the_file)

#################
### Helpers
#################

def pattern_match(string,
                  patterns):
    """
    Check to see if any substring in a list
    of patterns matches to a given string. Designed
    to be overly conservative for filtering out posts (i.e.
    matches substrings)

    Args:
        string (str): Input string
        patterns (iterable): Possible patterns
    
    Returns:
        match (bool): Whether or not a match exists
    """
    string_lower = string.lower()
    for p in patterns:
        if p in string_lower:
            return True
    return False

#################
### Classes
#################

class LoadProcessedData(object):

    """
    Generic Data Loading Class, for pre-processed data.
    """

    def __init__(self,
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
                 filter_empty=True,
                 emoji_handling=None,
                 filter_hashtag=False,
                 strip_hashtag=False,
                 max_tokens_per_document=None,
                 max_documents_per_user=None,
                 filter_mh_subreddits=None,
                 filter_mh_terms=None,
                 keep_retweets=True,
                 random_state=42):
        """
        Generic Data Loading Class

        Args:
            filter_negate (bool): Remove <NEGATE_FLAG> tokens
            filter_upper (bool): Remove <UPPER_FLAG> tokens
            filter_punctuation (bool): Remove standalone punctuation
            filter_numeric (bool): Remove <NUMERIC> tokens
            filter_user_mentions (bool): Remove <USER_MENTION> tokens
            filter_url (bool): Remove URL_TOKEN tokens
            filter_retweet (bool): Remove <RETWEET> and proceeding ":" tokens.
            filter_stopwords (bool): Remove stopwords from nltk english stopword set
            keep_pronouns (bool): If removing stopwords, keep pronouns
            preserve_case (bool): Keep token case as is. Otherwise, make lowercase.
            filter_empty (bool): Remove empty strings
            filter_hashtag (bool): If True and data loader encounters a hashtag, it will remove it
            strip_hashtag (bool): If True (default) and data loader encounters a hashtag with
                                  filter_hashtag set False, it will remove a hashtag prefix.
            emoji_handling (str or None): If None, emojis are kept as they appear in the text. Otherwise,
                                          should be "replace" or "strip". If "replace", they are replaced
                                          with a generic "<EMOJI>" token. If "strip", they are removed completely.
            max_tokens_per_document (int or None): Only consider the first N tokens from a single document.
                                                   If None (default), will take all tokens.
            max_documents_per_user (int or None): Only consider the most recent N documents from a user. If
                                                  None (default), will take all documents.
            filter_mh_subreddits (None or str): If None, no filtering. Otherwise either "depression","rsdd","smhd", or "all"
            filter_mh_terms (None or str): If None, no filtering. Otherwise either "rsdd" or "smhd"
            keep_retweets (bool): If True (default), keeps retweets in processed data. Otherwise,
                                  ignores them in the user's data.
            random_state (int): Seed to use for any random sampling
        """
        ## Class Arguments
        self.filter_negate = filter_negate
        self.filter_upper = filter_upper
        self.filter_punctuation = filter_punctuation
        self.filter_numeric = filter_numeric
        self.filter_user_mentions = filter_user_mentions
        self.filter_url = filter_url
        self.filter_retweet = filter_retweet
        self.filter_stopwords = filter_stopwords
        self.keep_pronouns = keep_pronouns
        self.preserve_case = preserve_case
        self.filter_empty = filter_empty
        self.emoji_handling = emoji_handling
        self.filter_hashtag = filter_hashtag
        self.strip_hashtag = strip_hashtag
        self.filter_mh_subreddits = filter_mh_subreddits
        self.filter_mh_terms = filter_mh_terms
        self.max_tokens_per_document = max_tokens_per_document
        self.max_documents_per_user = max_documents_per_user
        self.keep_retweets = keep_retweets
        self.random_state = random_state
        ## Helpful Variables
        self._punc = set()
        if self.filter_punctuation:
            self._punc = set(punctuation)
        ## Initialization Processes
        self._initialize_filter_set()
        self._initialize_stopwords()
        self._initialize_mh_subreddit_filter()
        self._initialize_mh_terms_filter()
    
    def _initialize_mh_subreddit_filter(self):
        """
        Helper. Initialize the mental-health subreddit filter
        set.

        Args:
            None
        
        Returns:
            None. Sets _ignore_subreddits attribute
        """
        ## Initalize Set of Subreddits to Ignore
        self._ignore_subreddits = set()
        ## Break If No Filtering Specified
        if not hasattr(self, "filter_mh_subreddits") or self.filter_mh_subreddits is None:
            return
        ## Check That Filter Exists
        if self.filter_mh_subreddits not in MH_SUBREDDITS:
            raise KeyError(f"Mental Health Subreddit Filter `{self.filter_mh_subreddits}` not found.")
        ## Update Ignore Set
        self._ignore_subreddits = set([i.lower() for i in MH_SUBREDDITS[self.filter_mh_subreddits]])

    def _initialize_mh_terms_filter(self,
                                    use_mh=True,
                                    use_pos=False,
                                    use_neg=False):
        """
        Helper. Initialize the list of terms to include
        in a set for filtering out posts. Consults
        the class initialization parameter filter_mh_terms
        for choosing the set of terms

        Args:
            use_mh (bool): If True, include the "terms" from
                           the mental-health term dictionary
            use_pos (bool): If True, include positive diagnosis
                           patterns from the dictionary
            use_neg (bool): If True, include negative diagnosis
                           patterns from the dictionary
        
        Returns:
            None, sets _ignore_terms attribute
        """
        ## Initialize Set of Terms to Ignore
        self._ignore_terms = set()
        ## Break if No Filtering Specified
        if not hasattr(self, "filter_mh_terms") or self.filter_mh_terms is None:
            return
        ## Check That Filter Exists
        if self.filter_mh_terms not in MH_TERMS["terms"]:
            raise KeyError(f"Mental Health Term Filter `{self.filter_mh_terms}` not found.")
        ## Construct Patterns
        all_patterns = []
        if use_mh:
            all_patterns.extend(MH_TERMS["terms"][self.filter_mh_terms])
        psets = []
        if use_pos:
            psets.append(MH_TERMS["pos_patterns"][self.filter_mh_terms])
        if use_neg:
            psets.append(MH_TERMS["neg_patterns"][self.filter_mh_terms])
        for pset in psets:
            for p in pset:
                if "_" in p:
                    p_sep = p.split()
                    n = len(p_sep) - 1
                    exp_match = [i for i, j in enumerate(p_sep) if j.startswith("_")][0]
                    exp_match_fillers = MH_TERMS["expansions"][p_sep[exp_match].rstrip(",")]
                    for emf in exp_match_fillers:
                        if p_sep[exp_match].endswith(","):
                            emf += ","
                        if exp_match != n:
                            emf_pat = " ".join(p_sep[:exp_match] + [emf] + p_sep[min(n, exp_match+1):])
                        else:
                            emf_pat = " ".join(p_sep[:exp_match] + [emf])
                        all_patterns.append(emf_pat)
                else:
                    all_patterns.append(p)
        self._ignore_terms = set(all_patterns)

    def _initialize_stopwords(self):
        """
        Initialize stopword set and removes pronouns if desired.

        Args:
            None
        
        Returns:
            None
        """
        ## Format Stopwords into set
        if hasattr(self, "filter_stopwords") and self.filter_stopwords:
            self.stopwords = set(STOPWORDS)
        else:
            self.stopwords = set()
            return
        ## Contraction Handling
        self.stopwords = self.stopwords | set(self._expand_contractions(list(self.stopwords)))
        ## Pronoun Handling
        if hasattr(self, "keep_pronouns") and self.keep_pronouns:
            for pro in PRONOUNS:
                if pro in self.stopwords:
                    self.stopwords.remove(pro)

    def _strip_emojis(self,
                      tokens):
        """
        Remove emojis from a list of tokens

        Args:
            tokens (list): Tokenized text
        
        Returns:
            tokens (list): Input list without emojis
        """
        tokens = list(filter(lambda t: t not in emoji.UNICODE_EMOJI and t != "<EMOJI>", tokens))
        return tokens
    
    def _replace_emojis(self,
                        tokens):
        """
        Replace emojis with generic <EMOJI> token

        Args:
            tokens (list): Tokenized text
        
        Returns:
            tokens (list): Tokenized text with emojis replaced with generic token
        """
        tokens = list(map(lambda t: "<EMOJI>" if t in emoji.UNICODE_EMOJI else t, tokens))
        return tokens

    def _strip_hashtags(self,
                        tokens):
        """
        
        """
        tokens = list(map(lambda t: t.replace("<HASHTAG=","")[:-1] if t.startswith("<HASHTAG=") else t, tokens))
        return tokens
    
    def _remove_hashtags(self,
                         tokens):
        """

        """
        tokens = list(filter(lambda t: not t.startswith("<HASHTAG="), tokens))
        return tokens

    def _expand_contractions(self,
                             tokens):
        """
        Expand English contractions.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now with expanded contractions.
        """
        tokens = \
        flatten(list(map(lambda t: CONTRACTIONS[t.lower()].split() if t.lower() in CONTRACTIONS else [t],
                         tokens)))
        return tokens
    
    def _select_n_recent_documents(self,
                                   user_data):
        """
        Select the N most recent documents in user data based on the 
        class initialization parameters.

        Args:
            user_data (list of dict): Processed user data dictionaries
        
        Returns:
            user_data (list of dict): N most recent documents
        """
        ## Downsample Documents
        user_data = sorted(user_data, key = lambda x: x["created_utc"], reverse = True)
        if hasattr(self, "max_documents_per_user") and self.max_documents_per_user is not None:
            user_data = user_data[:min(len(user_data), self.max_documents_per_user)]
        return user_data
    
    def _select_documents_in_date_range(self,
                                        user_data,
                                        min_date=None,
                                        max_date=None):
        """
        Isolate documents in a list of processed user_data
        that fall into a given date range

        Args:
            user_date (list of dict): Processed user data, includes
                                      created_utc key
            min_date (datetime or None): Lower date boundary
            max_date (datetime or None): Upper date boundary
        
        Returns:
            user_data (list of dict): Filtered user data
        """
        ## No additional filtering
        if min_date is None and max_date is None:
            return user_data
        ## Retrive Filtered Data
        filtered_data = []
        for u in user_data:
            tstamp = datetime.fromtimestamp(u["created_utc"])
            if min_date is not None and tstamp < min_date:
                continue
            if max_date is not None and tstamp > max_date:
                continue
            filtered_data.append(u)
        return filtered_data
    
    def _select_documents_randomly(self,
                                   user_data,
                                   n_samples=None):
        """
        Select a random sample of documents from an input
        user_data list

        Args:
            user_data (list of dict): Processed user data
            n_samples (int, float, or None): Max number of samples to select. If a float < 1,
                                            will sample percentage of user data available.
        
        Returns:
            random_sample (list of dict): Sampled user data
        """
        ## Null Samples
        if n_samples is None:
            return user_data
        ## Improper Spec
        if n_samples <= 0:
            raise ValueError("Cannot specify n_samples=0")
        ## Reset Random State
        if hasattr(self, "random_state"):
            np.random.seed(self.random_state)
        ## Sample Documents Without Replacement
        if n_samples < 1:
            n_samples = int(len(user_data) * n_samples)
        random_sample = list(np.random.choice(user_data,
                                              size=min(len(user_data), n_samples),
                                              replace=False))
        return random_sample
    
    def load_user_data(self,
                       filename,
                       min_date=None,
                       max_date=None,
                       n_samples=None,
                       randomized=False,
                       ):
        """
        Load preprocessed user data from disk.

        Args:
            filename (str): Path to .tar.gz file. Pre-processed
                            user data
            min_date (datetime or None): Lower date boundary
            max_date (datetime or None): Upper date boundary
            n_samples (int or None): Number of samples to consider
            randomized (bool): If sampling and set True, use a random
                               sample instead of the most recent posts

        Returns:
            user_data (list of dict): Preprocessed, filtered user
                                      data
        """
        ## Load the GZIPed file
        with gzip.open(filename) as the_file:
            user_data = json.load(the_file)
        ## Data Amount Filtering
        user_data = self._select_n_recent_documents(user_data)
        ## Date-based Filtering
        user_data = self._select_documents_in_date_range(user_data,
                                                         min_date,
                                                         max_date)
        ## Post-level Sampling
        if n_samples is not None:
            if randomized:
                user_data = self._select_documents_randomly(user_data,
                                                            n_samples)
            else:
                user_data = user_data[:min(len(user_data), n_samples)]
        ## Apply Processing
        user_data = self.filter_user_data(user_data)
        return user_data
    
    def load_user_metadata(self,
                           filename):
        """
        Load user metedata file

        Args:
            filename (str): Path to .tar.gz file
        
        Returns:
            user_data (dict): User metadata dictionary
        """
        ## Load the GZIPed file
        with gzip.open(filename) as the_file:
            user_data = json.load(the_file)
        return user_data
    
    def _filter_in(self,
                   obj,
                   ignore_set):
        """
        Filter a list by excluding matches in a set

        Args:
            obj (list): List to be filtered
            ignore_set (iterable): Set to check items again
        
        Returns:
            filtered_obj (list): Original list excluding objects
                          found in the ignore_set
        """
        return list(filter(lambda l: l not in ignore_set, obj))
    
    def _initialize_filter_set(self):
        """
        Initialize the set of items to filter from
        a tokenized text list based on class initialization
        parameters.

        Args:
            None
        
        Returns:
            None, assigns self.filter_set attribute
        """
        ## Initialize SEt
        self.filter_set = set()
        if hasattr(self,"filter_negate") and self.filter_negate:
            self.filter_set.add("<NEGATE_FLAG>")
        ## Filter Upper
        if hasattr(self,"filter_upper") and self.filter_upper:
             self.filter_set.add("<UPPER_FLAG>")
        ## Filter Numeric
        if hasattr(self,"filter_numeric") and self.filter_numeric:
           self.filter_set.add("<NUMERIC>")
        ## Filter User Mentions
        if hasattr(self,"filter_user_mentions") and self.filter_user_mentions:
            self.filter_set.add("<USER_MENTION>")
        ## Filter URL
        if hasattr(self,"filter_url") and self.filter_url:
            self.filter_set.add("<URL_TOKEN>")
        ## Filter Empty Strings
        if hasattr(self,"filter_empty") and self.filter_empty:
            self.filter_set.add("''")
            self.filter_set.add('""')
    
    def filter_user_data(self,
                         user_data):
        """
        Filter loaded user data based on class initialization
        parameters.

        Args:
            user_data (list of dict): Preprocessed user data
        
        Returns:
            filtered_data (list of dict): Filtered user data
        """
        ## Tokenized Text Field
        tt = "text_tokenized"
        ## Initialize Filtered Data Cache
        filtered_data = []
        for i, d in enumerate(user_data):
            ## Filter Based on Retweets
            if hasattr(self, "keep_retweets") and not self.keep_retweets and "<RETWEET>" in set(d["text_tokenized"]):
                continue
            ## Filter Based on Subreddit
            if hasattr(self, "filter_mh_subreddits") and self.filter_mh_subreddits is not None and "subreddit" in d.keys():
                if d["subreddit"].lower() in self._ignore_subreddits:
                    continue
            ## Filter Based on Terms
            if hasattr(self, "filter_mh_terms") and self.filter_mh_terms is not None:
                if "text" in d.keys():
                    if pattern_match(d["text"], self._ignore_terms):
                        continue
                else:
                    if pattern_match(" ".join(d["text_tokenized"]), self._ignore_terms):
                        continue
            ## Filter Based on Ignore Set
            d[tt] = self._filter_in(d[tt], self.filter_set)
            ## Length Check
            if len(d[tt]) == 0:
                filtered_data.append(d)
                continue
            ## Filter Retweet Tokens
            if hasattr(self, "filter_retweet") and self.filter_retweet and d[tt][0] == "<RETWEET>":
                if len(d[tt]) <= 1:
                    continue
                d[tt] = d[tt][1:]
                for _ in range(2):
                    if len(d[tt]) == 0:
                        break
                    if d[tt][0] in ["<USER_MENTION>", ":"]:
                        d[tt] = d[tt][1:]
            if hasattr(self, "filter_retweet") and self.filter_retweet:
                d[tt] = list(filter(lambda i: i!="<RETWEET>", d[tt]))
            ## Filter Hashtags
            if hasattr(self, "filter_hashtag") and self.filter_hashtag:
                d[tt] = self._remove_hashtags(d[tt])
            else:
                if hasattr(self, "strip_hashtag") and self.strip_hashtag:
                    d[tt] = self._strip_hashtags(d[tt])
            ## Max Tokens
            if hasattr(self, "max_tokens_per_document") and self.max_tokens_per_document is not None:
                d[tt] = d[tt][:min(len(d[tt]), self.max_tokens_per_document)]
            ## Filter Stopwords
            if hasattr(self, "filter_stopwords") and self.filter_stopwords:
                d[tt] = list(filter(lambda x: x.lower().replace("not_","") not in self.stopwords, d[tt]))
            ## Filter Punctuation
            if hasattr(self, "filter_punctuation") and self.filter_punctuation:
                d[tt] = list(filter(lambda i: not all(char in self._punc for char in i), d[tt]))
            ## Case Formatting
            if hasattr(self, "preserve_case") and not self.preserve_case:
                d[tt] = list(map(lambda i: "<HASHTAG={}".format(i.replace("<HASHTAG=","").lower()) if i.startswith("<HASHTAG=") else i, d[tt]))
                d[tt] = list(map(lambda tok: tok.lower() if tok not in self.filter_set and not tok.startswith("<HASHTAG") else tok, d[tt]))
            ## Emoji Handling
            if hasattr(self, "emoji_handling") and self.emoji_handling is not None:
                if self.emoji_handling == "replace":
                    d[tt] = self._replace_emojis(d[tt])
                elif self.emoji_handling == "strip":
                    d[tt] = self._strip_emojis(d[tt])
                else:
                    raise ValueError("emoji_handling should be 'replace', 'strip', or None.")
            filtered_data.append(d)
        return filtered_data


