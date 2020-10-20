
##################
### Imports
##################

## Standard Library
import os

## External Libraries
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler,
                                   normalize)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

## Local Modules
from ..util.logging import initialize_logger

## Create Logger
logger = initialize_logger()

##################
### Transformers
##################

class DummyTransformer(object):
    
    """
    Dummy Transformer. Methods of a Transfomer
    without any of the actual work.
    """

    def __repr__(self):
        """
        Description of the class.

        Args:
            None
        
        Returns:
            desc (str): Human readable string for the class.
        """
        return "DummyTransformer()"
    
    def fit(self,
            X,
            y=None):
        """
        Dummy fit method.

        Args:
            X (2d-array): Input matrix
            y (None): Ignored
        
        Returns:
            self
        """
        return self
    
    def transform(self,
                  X):
        """
        Dummy transform method.

        Args:
            X (2d-array): Input matrix
        
        Returns:
            X (2d-array): Input matrix, unchanged.
        """
        return X
    
    def fit_transform(self,
                      X,
                      y=None):
        """
        Apply dummy fit and transform methods

        Args:
            X (2d-array): Input feature matrix
            y (None): Ignored
        
        Returns:
            X (2d-array): Input matrix, unchanged
        """
        self = self.fit(X, y)
        X_T = self.transform(X)
        return X_T
        

class LIWCTransformer(object):

    """
    Created on Apr 5, 2013 by @author: luamct. Modified to support 
    easy conversion from a document term matrix to a LIWC 
    reprsentation.
    """

    def __init__(self,
                 vocab,
                 norm="max"):
        """
        Linquistic Inquiry and Word Count Transformer. Used to
        convert a document-term matrix into a LIWC representation.

        Args:
            vocab (Vocabulary class): Learned vocabulary
            norm (str): Either "matched", "unmatched", "max", "l1", or "l2", or None.
        """
        ## Vocabulary
        self.vocab = vocab
        ## Transformation Parameters
        self._norm = norm
        ## LIWC Variables
        self.dimensions = {}
        self.names = []
        self.entries = []
        self.cache = {}
        ## LIWC Dictionary Initialization
        self._initialize_dictionary()
    
    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        if not hasattr(self, "vocab"):
            self.vocab = None
        if not hasattr(self, "_norm"):
            self._norm = None
        desc = f"LIWCTransformer(vocab={self.vocab}, norm={self._norm})"
        return desc
    
    def _initialize_dictionary(self):
        """
        Initialize the LIWC dictionary as a python object.

        Args:
            None
        
        Returns:
            None
        """
        ## Get Path and Check for Existences
        liwc_path = "/".join(__file__.split("/")[:-1]) + \
                    f"/../../data/resources/liwc2007.dic"
        if not os.path.exists(liwc_path):
            raise FileNotFoundError(f"Could not find LIWC dictionary: {liwc_path}")
        logger.info("Loading LIWC Dictionary")
        ## Open the File
        _file = open(liwc_path,'r')
        # First read dimensions
        for line in _file:
            if line.strip()=='%':
                break
            # Get the index, ignore the name
            dim_id, dim_name = line.split()
            self.dimensions[dim_id] = len(self.dimensions)
            self.names.append(dim_name)
        # Now read word dictionary
        for line in _file:
            word_and_dims = line.split()
            word = word_and_dims[0]
            dims = word_and_dims[1:]
            if (word[-1] != '*') :
                self.cache[word] = dims
            else :
                self.entries.append( (word, dims) ) 
        _file.close()

    def matches(self,
                word,
                entry):
        """
        Check to see if a word matches an entry in the LIWC dictionary.

        Args:
            word (str): Input word
            entry (str): String in the LIWC dictionary
        
        Returns:
            is_match (bool): Input word starts with LIWC entry
        """
        is_match = word.startswith(entry[:-1])
        return is_match

    def add(self,
            count,
            dimensions) :
        """
        Add a match count of a count dictionary.

        Args:
            count (1d-array): Counts by LIWC dimension
            dimensions (dict): Dimension mapping
        
        Returns:
            None, updates count in place.
        """
        # Get the dimension index and increment
        for dim_id in dimensions:
            dim_index = self.dimensions[dim_id]
            count[dim_index] += 1

    def search(self,
               word):
        """
        Have to iterate through every word because there are also regular expressions,
        otherwise we could have used a set or hashset structure for efficiency.

        Args:
            word (str): See if a word matches any dimensions in the LIWC dictionary.

        Returns:
            dimensions (list or None): Dimensions a word matches to.
        """
        for entry, dimensions in self.entries :
            if self.matches(word, entry) :
                return dimensions
        return None

    def binary(self,
               word):
        """
        Search for a word match across the entries using binary search

        Args:
            word (str): Input word
        
        Returns:
            matches (list or None): Recursive call of _binary_ for the word.
        """
        return self._binary_(word, 0, len(self.entries))

    def _binary_(self,
                 word,
                 s,
                 e):
        """
        Recursive function checking for word matches in the LIWC dictionary.

        Args:
            word (str): Input word
            s (int): Start index
            e (int): End index
        
        Returns:
            matches (list or None): Entries if the word matches to an entry
                                    in the LIWC dictionary.
        """
        if s >= e:
            return None
        m  = int((s+e)/2)
        if m < len(self.entries):
            mword = self.entries[m][0][:-1] # Remove * at the end
            # Matched
            if word.startswith(mword):
                return self.entries[m][1]
            # Search in the right partition
            if word < mword :
                return self._binary_(word, s, m)
            if word > mword :
                return self._binary_(word, m+1, e)
        return None

    def classify(self,
                 words):
        """
        Counts the occurrence of each dimension in the input words
        Use 2 extra positions for the total of words considered and the
        total of words successfully matched in the LIWC dictionary

        Args:
            words (list of str): Individual tokens
        
        Returns:
            n_words (int): Length of input list
            tokens_matched (int): Number of tokens matched to dictionary
            count (1d-array): Matched per LIWC dimension
        """
        count = np.zeros(len(self.dimensions), dtype=int)
        tokens_matched=0
        for word in words:
            # If not in cache, look for it and add it if found
            if word not in self.cache :
                dimensions = self.binary(word)
                if dimensions != None :
                    self.cache[word] = dimensions
            # If still not in cache then it's not in the LIWC dictionary
            if word in self.cache :
                self.add(count, self.cache[word])
                tokens_matched += 1
        return (len(words), tokens_matched, count)
    
    def fit(self,
            X,
            y=None):
        """
        Learns the dimension mapping between each vocabulary term and a LIWC dim.
        
        Args:
            X (2d-array): Document Term Matrix
            y (None): Ignored
        
        Returns:
            self
        """
        ## Get Ordered Vocabulary
        ordered_vocabulary = self.vocab.get_ordered_vocabulary()
        ## Isolate Unigrams
        ordered_vocabulary = list(map(lambda v: list(v)[0] if len(v) == 1 else "<LIWC_IGNORE_WORD>", ordered_vocabulary))
        ## Get Mapping Between LIWC Dimensions and Vocabulary
        self._dim_map = []
        for o in ordered_vocabulary:
            if o == "<LIWC_IGNORE_WORD>":
                self._dim_map.append(np.zeros(64))
            else:
                _, _, word_dims = self.classify([o.lower()])
                self._dim_map.append(word_dims)
        self._dim_map = np.vstack(self._dim_map)
        return self
    
    def transform(self,
                  X):
        """
        Transform a document-term matrix into a LIWC representation

        Args:
            X (2d-array): Document-term matrix
        
        Returns:
            X_T (2d-array): LIWC representation
        """
        ## Tranform to Dense
        if isinstance(X, csr_matrix):
            X = X.toarray()
        ## Get Transformed Rep
        X_T = np.matmul(X, self._dim_map)        
        ## Normalize
        if hasattr(self, "_norm") and self._norm is not None and \
           self._norm != "unmatched" and \
           self._norm != "matched":
            X_T = normalize(X_T, norm = self._norm, axis=1)
        elif hasattr(self, "_norm") and self._norm == "unmatched":
            norm = X.sum(axis=1,keepdims=True)
            X_T = np.divide(X_T,
                            norm,
                            out = np.zeros_like(X_T),
                            where = norm > 0)
        elif hasattr(self, "_norm") and self._norm == "matched":
            vocab_matches = vocab_matches = (self._dim_map > 0).any(axis=1).reshape(-1,1).astype(int)
            norm = np.matmul(X, vocab_matches)
            X_T = np.divide(X_T,
                            norm,
                            out = np.zeros_like(X_T),
                            where = norm > 0)
        return X_T
    
    def fit_transform(self,
                      X,
                      y=None):
        """
        Learns the dimension mapping between each vocabulary term and a LIWC dim.
        Then, apply the learned transformation to the input feature matrix X.

        Args:
            X (2d-array): Document Term Matrix
            y (None): Ignored

        Returns:
            X_T (2d-array): LIWC representation
        """
        ## Learn Dimension Mapping
        self = self.fit(X)
        ## Apply Transformation
        X_T = self.transform(X)
        return X_T


class EmbeddingTransformer(object):
    
    """
    Used to transform a count-based document-term matrix into
    an embedding representation.
    """

    def __init__(self,
                 vocab,
                 dim = 25,
                 pooling = "mean",
                 jobs = 8):
        """
        Document-Term to Document-Embedding Matrix

        Args:
            vocab (Vocabulary): Class with a learned vocabulary.
            dim (int): [25, 50, 100, 200]. Dimensionality of GloVe embeddings to use.
            pooling (str): Either "mean" or "max". Default is "mean". "max" actually 
                           chooses the extrema of the dimension.
            jobs (int): Number of processes to use when multiprocessing is available
        """
        self.dim = dim
        self.pooling = pooling
        self.vocab = vocab
        self.jobs = jobs
        ## Initialize Class
        _ = self._validate_params()
        _ = self._load_embeddings()
    
    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        if not hasattr(self,"vocab"):
            self.vocab = None
        if not hasattr(self, "pooling"):
            self.pooling = None
        if not hasattr(self, "dim"):
            self.dim = None
        desc = f"EmbeddingTransform(vocab={self.vocab}, pooling={self.pooling}, dim={self.dim})"
        return desc

    def _validate_params(self):
        """
        Check that the pooling method chosen is supported.

        Args:
            None
        
        Returns:
            None
        
        Raises:
            ValueError: if self.pooling is not == "mean" or "max"
        """
        if not hasattr(self, "pooling") or self.pooling not in ["mean","max"]:
            raise ValueError("pooling must be one of `max` or `mean`")
        if not hasattr(self, "dim") or self.dim not in [25,50,100,200]:
            raise ValueError("Dimension must be in [25,50,100,200]")
    
    def _process_embedding_line(self,
                                line):
        """
        Identify a token and vector on a line of 
        a pretrained embeddings file

        Args:
            line (str): Line from embeddings file
        
        Returns:
            tok (str): Embedding token
            vector (1d-array): Embedding vector
        """
        ## Ignore Special Characters
        if line.startswith("<"):
            return (None, None)
        ## Inital Split
        tok, emb = line.split(" ", 1)
        ## Separate Token and Embedding
        emb = np.array(emb.split()).astype(float)
        ## Hashtag Handling
        if tok == "#":
            return (None, None)
        if tok.startswith("#"):
            tok = "<HASHTAG={}>".format(tok)
        ## Tuple Formatting
        tok = (tok, )
        ## Return Based on Vocab
        if tok in self.vocab.vocab:
            return (tok, emb)
        else:
            return (None, None)

    def _load_embeddings(self):
        """
        Load an embeddings matrix, confined to the vocabulary passed
        to the class during initialization.

        Args:
            None
        
        Returns:
            None, initializes embeddings
        """
        ## Get Path and Check for Existences
        embeddings_path = "/".join(__file__.split("/")[:-1]) + \
                          f"/../../data/resources/glove.twitter.27B.{self.dim}d.txt"
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Could not find embeddings: {embeddings_path}")
        ## Load Embeddings Dictionary
        logger.info("Loading Embeddings")
        with open(embeddings_path, "r") as the_file:
            word2emb = dict()
            for line in the_file:
                tok, emb = self._process_embedding_line(line)
                if tok is not None:
                    word2emb[tok] = emb
        self._matched_tokens = list(word2emb.keys())
        ## Sort Embeddings into a Matrix
        self.embedding_matrix = np.zeros((len(self.vocab.vocab), self.dim))
        ordered_vocab = self.vocab.get_ordered_vocabulary()
        for i, o in enumerate(ordered_vocab):
            if o not in word2emb:
                continue
            self.embedding_matrix[i] = word2emb[o]
    
    def fit(self,
            X,
            y=None):
        """
        Does nothing to input, required for sklearn pipeline.

        Args:
            X (2d-array): Input document-term matrix
            y (None): Ignored entirely.
        
        Returns:
            self
        """
        return self
    
    def _absmaxND(self,
                  a,
                  axis=None):
        """
        Get the extrema of a dimension in an array.

        Args:
            a (2d-array): Input array
            axis (int or None): Array to perform extrema check on.
        
        Returns:
            extrema (array): Extrema along the specified axis
        """
        amax = a.max(axis)
        amin = a.min(axis)
        return np.where(-amin > amax, amin, amax)

    def transform(self,
                  X):
        """
        Transform a document-term matrix into the embedding space

        Args:
            X (2d-array): Document-term matrix (ordering of the vocabulary)
        
        Returns:
            X_T (2d-array): Document-term matrix in the embedding space
        """
        ## Transform to Dense Array
        if isinstance(X, csr_matrix):
            X = X.toarray()
        ## Case 1: Mean Pooling
        if hasattr(self, "pooling") and self.pooling == "mean":
            matched_tokens = [self.vocab.ngram_to_idx[i] for i in self._matched_tokens]
            matched_token_count = X[:, matched_tokens].sum(axis=1, keepdims=True)
            X_T = np.matmul(X, self.embedding_matrix)
            X_T = np.divide(X_T,
                            matched_token_count,
                            out=np.zeros((X.shape[0], self.dim)),
                            where=matched_token_count!=0)
            return X_T
        elif hasattr(self, "pooling") and self.pooling == "max":
            nonzero_elements = list(map(lambda x: np.nonzero(x)[0], X))
            X_T = list(map(lambda x: self._absmaxND(self.embedding_matrix[x], axis=0) if len(x) > 0 else np.zeros(self.embedding_matrix.shape[1]), nonzero_elements))
            X_T = np.vstack(X_T)
            return X_T
    
    def fit_transform(self,
                      X):
        """
        Apply fit() and transform() methods to input X

        Args:
            X (2d-array): Document Term matrix
        
        Returns:
            X_T (2d-array): Matrix now in the embedding space.
        """
        _ = self.fit(X)
        X_T = self.transform(X)
        return X_T

##################
### Pipeline
##################

## Transformers (Class, Requires Vocab)
_transformer_dict = {
            "bag_of_words":(DummyTransformer, False),
            "tfidf":(TfidfTransformer, False),
            "liwc":(LIWCTransformer, True),
            "glove":(EmbeddingTransformer, True),
            "lda":(LatentDirichletAllocation, False),           
}


class FeaturePreprocessor(object):

    """
    Wrapper for feature preprocessing
    """

    def __init__(self,
                 vocab,
                 feature_flags={},
                 feature_kwargs={},
                 standardize=True):
        """
        Wrapper for Feature Preprocessing pipeline.

        Args:
            vocab (Vocabulary): Input feature vocabulary
            feature_flags (dict): Which features should be used and which should not
            feature_kwargs (dict): Parameters for each of the types of features
            standardize (bool): Whether or not to standardize the feature matrix
        """
        ## Class Attributes
        self.vocab = vocab
        self._feature_flags = feature_flags
        self._feature_kwargs = feature_kwargs
        self._standardize = standardize
        ## Class Variable Workspace
        self._transformers = {}
        self._transformer_names = []
        ## Initialization
        _ = self._validate_preprocessing_params()
        _ = self._initialize_preprocessing_pipline()
    
    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        desc = f"FeaturePreprocessor(vocab={self.vocab}, feature_flags={self._feature_flags}, feature_kwargs={self._feature_kwargs})"
        return desc

    def _validate_preprocessing_params(self):
        """
        Check that the input preprocessing arguments are a valid combination

        Args:
            None
        
        Returns:
            None
        
        Raises:
            ValueError: Errors based on invalid combinations of preprocessing kwargs
        """
        ## Update Initialization Defaults
        if self._feature_kwargs is None:
            self._feature_kwargs = {}
        if self._standardize is None:
            self._standardize = False
        ## GloVe Embeddings Check
        if "glove" in self._feature_flags and self._feature_flags.get("glove"):
            glove_args = {}
            if "glove" in self._feature_kwargs:
                glove_args = self._feature_kwargs.get("glove")
            if "dim" in glove_args and not isinstance(glove_args["dim"], int):
                raise ValueError("Expected `glove` dimension param to be an int.")
            if "dim" in glove_args and glove_args["dim"] not in [25,50,100,200]:
                raise ValueError("`glove` dimension param must be in [25,50,100,200]")
        ## LIWC Check
        if "liwc" in self._feature_flags and self._feature_flags.get("liwc"):
            liwc_args = {}
            if "liwc" in self._feature_kwargs:
                liwc_args = self._feature_kwargs.get("liwc")
            if "norm" in liwc_args and \
               liwc_args["norm"] is not None and \
               not isinstance(liwc_args["norm"], str):
               raise ValueError("Expected `norm` in liwc kwargs to be None or a str.")

    def _initialize_transformer(self,
                                transformer_key):
        """
        Initialize the transformer objects depending on their on/off specification
        in feature_flags argument passed to class initialization

        Args:
            transformer_key (str): name of the transformer
        
        Returns:
            None, initializes _transformers dict in place
        """
        if transformer_key in self._feature_flags and self._feature_flags.get(transformer_key):
            _kwargs = self._feature_kwargs.get(transformer_key) if transformer_key in self._feature_kwargs else {}
            if _transformer_dict[transformer_key][1]:
                _kwargs["vocab"] = self.vocab
            self._transformers[transformer_key] = _transformer_dict[transformer_key][0](**_kwargs)


    def _initialize_preprocessing_pipline(self):
        """
        Initialize the sklearn Pipeline object based on 
        class preprocessing parameters.

        Args:
            None
        
        Returns:
            None, sets self._pipeline in place.
        """
        ## Check Attribute
        if not hasattr(self, "_standardize"):
            self._standardize = False
        ## Initialize Preprocessing Transformers
        for transformer_key in _transformer_dict.keys():
            self._initialize_transformer(transformer_key)
        ## Standardization
        if self._standardize:
            self._transformers["standardize"] = StandardScaler()
        ## Construct Pipeline
        self._transformer_names = sorted([i for i in self._transformers.keys() if i != "standardize"])
        logger.info("Using the following feature set ({}): {}".format(
            {True:"w/ Standardization", False:"w/o Standardization"}[self._standardize],
            self._transformer_names))

    def fit(self,
            X,
            y=None):
        """
        Fit the preprocessing pipeline using input feature matrix.

        Args:
            X (2d-array): Input feature matrix
            y (None): Ignored
        
        Returns:
            self: Class with trained feature preprocessing pipeline.
        """
        ## Transform to Dense Array
        if isinstance(X, csr_matrix):
            X = X.toarray()
        ## Transformed Feature Cache
        X_T = []
        ## Fit Independent Transformers
        for i, t in enumerate(self._transformer_names):
            logger.info("Learning Feature Set ({}/{}): {}".format(
                    i+1, len(self._transformer_names), t
            ))
            self._transformers[t] = self._transformers[t].fit(X, y)
            X_T.append(self._transformers[t].transform(X))
        ## Concatenate Features
        X_T = np.hstack(list(map(lambda x: x.toarray() if isinstance(x, csr_matrix) else x, X_T)))
        ## Fit Standard Scaler
        if hasattr(self, "_standardize") and self._standardize:
            logger.info("Learning Standardization Parameters")
            self._transformers["standardize"] = self._transformers["standardize"].fit(X_T)
        return self
    
    def transform(self,
                  X):
        """
        Apply learned preprocessing pipeline to a feature matrix.

        Args:
            X (2d-array): Feature matrix, of the form of the data the class
                          fit() method was called on.
        
        Returns:
            X_T (2d-array): Transformed feature representation.
        """
        ## Transform to Dense Array
        if isinstance(X, csr_matrix):
            X = X.toarray()
        ## Transformed Feature Cache
        X_T = []
        ## Transform Document-Term Matrix
        for i, t in enumerate(self._transformer_names):
            X_t = self._transformers[t].transform(X)
            X_T.append(X_t)
        ## Concatenate
        X_T = np.hstack(list(map(lambda x: x.toarray() if isinstance(x, csr_matrix) else x, X_T)))
        ## Apply Standardization
        if hasattr(self, "_standardize") and self._standardize:
            X_T = self._transformers["standardize"].transform(X_T)
        return X_T
    
    def fit_transform(self,
                      X):
        """
        Fit the preprocessing pipeline using input feature matrix.
        Then, apply learned preprocessing pipeline to a feature matrix.

        Args:
            X (2d-array): Input feature matrix
            y (None): Ignored
        
        Returns:
            X_T (2d-array): Transformed feature representation.
        """
        ## Fit the Pipeline
        self = self.fit(X)
        ## Apply the Transformation
        X_T = self.transform(X)
        return X_T
