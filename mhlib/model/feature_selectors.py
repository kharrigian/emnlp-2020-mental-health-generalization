
##################
### Imports
##################

## External Libraries
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

## Local Modules
from ..preprocess.tokenizer import (STOPWORDS,
                                    PRONOUNS)
from ..util.logging import initialize_logger

## Create Logger
logger = initialize_logger()

##################
### Helpers
##################

def _get_rank_order(array):
    """
    Rank the elements of an array.

    Args:
        array (numpy 1d-array): Input array to rank
    
    Returns:
        rank (numpy 1d-array): Rank order, descending (e.g. 0 = indice of highest value)
    """
    order = array.argsort()
    rank = order.argsort()
    rank = len(rank) - 1 - rank
    return rank

##################
### Selectors
##################

class KLDivergenceSelector(object):

    """
    Method inspired by 'Non-localness' to reduce a vocabulary
    given a document-term representation in a classification
    problem
    """

    def __init__(self,
                 vocab,
                 top_k=50000,
                 min_support=30,
                 stopwords=None,
                 add_lambda=1,
                 beta=0.25,
                 symmetric=True,
                 preserve_protected=False,
                 filter_stopwords=False,
                 keep_pronouns=False):
        """
        Args:
            vocab (Vocabulary): Current vocabulary.
            top_k (int): Number of top scored vocabulary terms to filter the 
                         matrix down to.
            min_support (int): Minimum frequency across documents for a vocabulary
                               term to be kept.
            stopwords (iterable): Set of stopwords to use as part of the 
                                  dimensionality reduction.
            add_lambda (float): Smoothing parameter for the count-based
                                probabilities
            beta (float [0, 1]): Trade-off between term-frequency and 
                                 divergence score. Beta = 1 means we 
                                 rely only on term-frequency.
            symmetric (bool): If True (default), use symmetric KL divergence as measure
                              of distance. Otherwise, only use over-indexing on class
            preserve_protected (bool): If True, will not filter out words from vocabulary's
                                       protected term dictionary
            filter_stopwords (bool): If True, will force filter stopwords during vocabulary 
                                     update (regardless of whether they are protected)
            keep_pronouns (bool): If True, any filtering will preserve pronouns regardless
                                  of frequency or score
        """
        ## Class Attributes
        self.vocab = vocab
        self._stopwords = stopwords
        self._top_k = top_k
        self._min_support = min_support
        self._add_lambda = add_lambda
        self._beta = beta
        self._symmetric = symmetric
        self._preserve_protected = preserve_protected
        self._filter_stopwords = filter_stopwords
        self._keep_pronouns = keep_pronouns
        ## Class Initialization of Stopwords
        self._initialize_stopword_set()
        ## Workspace
        self._term_freq = None
        self._div_score = None
        self._vocabulary_terms = None
    
    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        desc = f"KLDivergenceSelector(top_k={self._top_k}, min_support={self._min_support}, add_lambda={self._add_lambda}, beta={self._beta})"
        return desc

    def _initialize_stopword_set(self):
        """
        Initialize the stopword set to be used as a reference distribution
        for making the dimensionality reduction.

        Args:
            None
        
        Returns:
            None, initializes stopword set based on class attributes.
        """
        logger.info("Initializing Stopword Set in KLDivergenceSelector")
        ## Check to See if Stopset is None
        if not hasattr(self, "_stopwords") or self._stopwords is None:
            self._stopwords = STOPWORDS
        ## Make a Set
        if not isinstance(self._stopwords, set):
            self._stopwords = set(self._stopwords)
        ## Expand Contractions
        self._stopwords = self._stopwords | set(self.vocab._loader._expand_contractions(list(self._stopwords)))
        ## Filter Stopset against Vocabulary
        vocabulary_terms = self.vocab.get_ordered_vocabulary()
        vocabulary_terms = list(map(lambda v: list(v)[0] if len(v) == 1 else "<TERM_TO_IGNORE>", vocabulary_terms))
        vocabulary_terms_lower = set(list(map(lambda i: i.lower().replace("not_",""), vocabulary_terms)))
        for s in list(self._stopwords):
            if s not in vocabulary_terms_lower:
                self._stopwords.remove(s)
        if len(self._stopwords) == 0:
            raise ValueError("Stopword set does not include any terms matched to the input vocabulary")
        
    def _update_vocabulary(self):
        """
        Update the vocabulary object after selecting a new subset of features
        to use in your model.

        Args:
            None
        
        Returns:
            None, updates vocabulary object in place.
        """
        logger.info("Updating Vocabulary")
        ## Identify Terms to Definitely Remove
        force_remove = set()
        if hasattr(self, "_filter_stopwords") and self._filter_stopwords:
            force_remove = STOPWORDS | set(self.vocab._loader._expand_contractions(list(STOPWORDS)))
            if hasattr(self, "_keep_pronouns") and self._keep_pronouns:
                force_remove = set([i for i in force_remove if i not in PRONOUNS])
        ## Identify Protected Terms
        protected = set()        
        if hasattr(self.vocab, "_protected_set"):
            for key, terms in self.vocab._protected_set.items():
                tkeep = [t for t in terms if t not in force_remove]
                protected.update(tkeep)
        if hasattr(self, "_keep_pronouns") and self._keep_pronouns:
            protected.update(list(PRONOUNS))
        ## Apply Filtering
        top_k_set = set(self.top_k) if hasattr(self, "top_k") else set()
        for term, ind in list(self.vocab.ngram_to_idx.items()):
            if "".join(term) in protected:
                continue
            if ind not in top_k_set or "".join(term) in force_remove:
                self.vocab.vocab.remove(term)
        new_ngram_to_idx = dict((term, i) for i, term in enumerate(sorted(self.vocab.vocab)))
        new_rev_ngram_to_idx = dict((y, x) for x, y in new_ngram_to_idx.items())
        self.top_k = []
        for index in sorted(new_rev_ngram_to_idx):
            self.top_k.append(self.vocab.ngram_to_idx[new_rev_ngram_to_idx[index]])
        self.vocab.ngram_to_idx = new_ngram_to_idx

    def fit(self,
            X,
            y):
        """
        Learn feature scores based on the class parameters

        Args:
            X (2d-array): Document Term Matrix
            y (1d-array): Class-labels
        
        Returns:
            self
        """
        logger.info("Learning KLDivergence Reduction")
        ## Transform to Dense
        if isinstance(X, csr_matrix):
            X = X.toarray()
        ## Binarize
        logger.info("Binarizing Document Term Matrix")
        X = (X.copy() > 0).astype(int)
        ## Compute Stopset Indices
        self._vocabulary_terms = self.vocab.get_ordered_vocabulary()
        self._vocabulary_terms = list(map(lambda v: list(v)[0] if len(v) == 1 else "<TERM_TO_IGNORE>", self._vocabulary_terms))
        self._vocabulary_terms = list(map(lambda i: i.lower(), self._vocabulary_terms))
        stopset_ind = dict()
        for i, v in enumerate(self._vocabulary_terms):
            v_clean = v.replace("not_","")
            if v_clean in self._stopwords:
                if v_clean not in stopset_ind:
                    stopset_ind[v_clean] = []
                stopset_ind[v_clean].append(i)
        ## Compute Generic Term Freq
        logger.info("Computing Term Frequencies")
        self._term_freq = X.sum(axis=0)
        ## Compute Generic Stopword Weights
        logger.info("Computing Stopword Weights")
        stop_freq = dict()
        for word, ind in stopset_ind.items():
            stop_freq[word] =  self._term_freq[ind].sum() + self._add_lambda
        stop_freq_sum = sum(stop_freq.values())
        stop_freq = dict((x, y/stop_freq_sum) for x, y in stop_freq.items())
        ## Get Class Indices
        class_ind = dict((i, np.where(y==i)[0]) for i in sorted(set(y)))
        ## Get P(class|feature)
        if not hasattr(self, "_add_lambda"):
            self._add_lambda = 1
        logger.info("Computing P(class|feature)")
        V = len(self._vocabulary_terms)
        p_c_f = np.zeros((len(class_ind), V))
        for c, ind in class_ind.items():
            p_c_f[c] += X[ind].sum(axis=0)
        p_c_f = (p_c_f + self._add_lambda) / (p_c_f + self._add_lambda).sum(axis=0)
        ## Compute Divergence Score
        logger.info("Computing Divergence Score")
        self._div_score = np.zeros((len(class_ind), V))
        for stopword, stop_ind in stopset_ind.items():
            p_c_stop = (p_c_f[:, stop_ind].sum(axis=1) / p_c_f[:, stop_ind].sum()).reshape(-1,1)
            if self._symmetric:
                sim_stopword_f = p_c_f * np.log(p_c_f / p_c_stop) +  p_c_stop * np.log(p_c_stop / p_c_f)
            else:
                sim_stopword_f = p_c_f * np.log(p_c_f / p_c_stop)
            self._div_score += stop_freq[stopword] * sim_stopword_f
        ## Compute Weighted Rank (including frequency)
        logger.info("Computing Weighted Rank (Divergence Score/Term Frequency)")
        div_score_rank = _get_rank_order(self._div_score.sum(axis=0))
        term_freq_rank = _get_rank_order(self._term_freq)
        if not hasattr(self, "_beta"):
            self._beta = 0
        weighted_rank_val = (1 - self._beta) * div_score_rank + self._beta * term_freq_rank
        weighted_rank = weighted_rank_val.argsort()
        ## Threshold Filtering
        logger.info("Selecting Top-K Terms")
        thres_filt = set(np.nonzero(self._term_freq >= self._min_support)[0])
        ## Get Top-K Scored Terms
        self.top_k = []
        count = 0
        for s in weighted_rank:
            if count == self._top_k:
                break
            if s in thres_filt:
                self.top_k.append(s)
                count += 1
        ## Update Vocabulary
        self._update_vocabulary()
        return self
    
    def transform(self,
                  X):
        """
        Perform feature reduction on a full document-term matrix based
        on the learned feature scores.

        Args:
            X (2d-array): Document Term Matrix
        
        Returns:
            X_red (2d-array): Reduced Dimensionality Document Term Matrix
        """
        X_red = X[:, self.top_k]
        return X_red
    
    def fit_transform(self,
                      X,
                      y):
        """
        Apply fit and transform methods to a document-term matrix 
        to get a reduced dimensionality representation.

        Args:
            X (2d-array): Document Term Matrix
            y (1d-array): Class labels
        
        Returns:
            X_red (2d-array): Reduced Dimensionality Document Term Matrix
        """
        ## Fit
        _ = self.fit(X, y)
        ## Transform
        X_red = self.transform(X)
        return X_red

class PMISelector(object):

    """
    Pointwise Mutual Information Selector
    """

    def __init__(self,
                 vocab,
                 top_k=30000,
                 min_support=10,
                 add_lambda=1,
                 beta=0.0,
                 binarize=False,
                 symmetric=True,
                 preserve_protected=False,
                 filter_stopwords=False,
                 keep_pronouns=False
                 ):
        """
        Pointwise Mututal Information Selector for Document-
        Term Matrices

        Args:
            vocab (Vocabulary): Vocabulary used to construct document-
                                term matrix
            top_k (int): Number of desired features
            min_support (int): Minimum frequency to keep feature
            add_lambda (float): Smoothing parameter
            beta (float [0,1]): How much to factor frequency into weighted
                                rank. 0 means None, 1 means entirely
            binarize (bool): Whether the document-term matrix should
                             be binarized prior to computing PMI
            symmetric (bool): Whether PMI values should be summed over
                              classes. False means only the positive class
                              is used to select features. Only relevant
                              if target classes are [0, 1].
            preserve_protected (bool): If True, will not filter out words from vocabulary's
                                       protected term dictionary
            filter_stopwords (bool): If True, will force filter stopwords during vocabulary 
                                     update (regardless of whether they are protected)
            keep_pronouns (bool): If True, any filtering will preserve pronouns regardless
                                  of frequency or score
        """
        ## Store Class Parameters
        self.vocab = vocab
        self._top_k = top_k
        self._min_support = min_support
        self._add_lambda = add_lambda
        self._beta = beta
        self._binarize = binarize
        self._symmetric = symmetric
        self._preserve_protected = preserve_protected
        self._filter_stopwords = filter_stopwords
        self._keep_pronouns = keep_pronouns
        ## Variable Working Space
        self._classes = []
        self._original_vocab = []
        self.pmi = None
        self.top_k = []

    def _update_vocabulary(self):
        """
        Update the vocabulary object after selecting a new subset of features
        to use in your model.

        Args:
            None
        
        Returns:
            None, updates vocabulary object in place.
        """
        logger.info("Updating Vocabulary")
        ## Identify Terms to Definitely Remove
        force_remove = set()
        if hasattr(self, "_filter_stopwords") and self._filter_stopwords:
            force_remove = STOPWORDS | set(self.vocab._loader._expand_contractions(list(STOPWORDS)))
            if hasattr(self, "_keep_pronouns") and self._keep_pronouns:
                force_remove = set([i for i in force_remove if i not in PRONOUNS])
        ## Identify Protected Terms
        protected = set()        
        if hasattr(self.vocab, "_protected_set"):
            for key, terms in self.vocab._protected_set.items():
                tkeep = [t for t in terms if t not in force_remove]
                protected.update(tkeep)
        if hasattr(self, "_keep_pronouns") and self._keep_pronouns:
            protected.update(list(PRONOUNS))
        ## Apply Filtering
        top_k_set = set(self.top_k) if hasattr(self, "top_k") else set()
        for term, ind in list(self.vocab.ngram_to_idx.items()):
            if "".join(term) in protected:
                continue
            if ind not in top_k_set or "".join(term) in force_remove:
                self.vocab.vocab.remove(term)
        new_ngram_to_idx = dict((term, i) for i, term in enumerate(sorted(self.vocab.vocab)))
        new_rev_ngram_to_idx = dict((y, x) for x, y in new_ngram_to_idx.items())
        self.top_k = []
        for index in sorted(new_rev_ngram_to_idx):
            self.top_k.append(self.vocab.ngram_to_idx[new_rev_ngram_to_idx[index]])
        self.vocab.ngram_to_idx = new_ngram_to_idx

    def fit(self,
            X,
            y):
        """
        Learn PMI within document-term matrix

        Args:
            X (2d-array): Document-term matrix
            y (1d-array): Class labels
        
        Returns:
            self
        """
        logger.info("Learning PMI Reduction")
        ## Transform to Dense Array
        if isinstance(X, csr_matrix):
            X = X.toarray()
        ## Format Labels
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        ## Identify Masks
        self._classes = sorted(set(y))
        masks = []
        for c in self._classes:
            mc = np.where(y == c)[0]
            masks.append(mc)
        ## Binarize (If Desired)
        if hasattr(self, "_binarize") and self._binarize:
            X = (X>0).astype(int)
        ## Vocabulary Size
        V = X.shape[1]
        ## Compute Feature Frequencies
        self.freq = X.sum(axis=0)
        self._original_vocab = self.vocab.get_ordered_vocabulary()
        ## Compute Baseline Feature Probabilites (With Smoothing)
        p_x = (X.sum(axis=0) + self._add_lambda) / (np.sum(X) + V * self._add_lambda)
        ## Store P(x|y) Values (With Smoothing)
        p_x_given_y = np.ones((len(masks), V)) * np.nan
        for i, (m, c) in enumerate(zip(masks, self._classes)):
            X_m = X[m]
            p_x_given_y[i] = (X_m.sum(axis=0) + self._add_lambda) / (np.sum(X_m) + V * self._add_lambda)
        ## Calculate PMI
        self.pmi = np.log2(p_x_given_y / p_x)
        ## Compute Weighted Rank (including frequency)
        logger.info("Computing Weighted Rank (Divergence Score/Term Frequency)")
        if not hasattr(self, "_symmetric") or self._symmetric or self._classes != [0,1]:
            pmi_score_rank = _get_rank_order(self.pmi.sum(axis=0))
        else:
            pmi_score_rank = _get_rank_order(self.pmi[1])
        term_freq_rank = _get_rank_order(self.freq)
        weighted_rank_val = (1 - self._beta) * pmi_score_rank + self._beta * term_freq_rank
        weighted_rank = weighted_rank_val.argsort()
        ## Threshold Filtering
        logger.info("Selecting Top-K Terms")
        thres_filt = set(np.nonzero(self.freq >= self._min_support)[0])
        ## Get Top-K Scored Terms
        self.top_k = []
        if not hasattr(self, "_top_k"):
            self._top_k = len(weighted_rank)
        count = 0
        for s in weighted_rank:
            if count == self._top_k:
                break
            if s in thres_filt:
                self.top_k.append(s)
                count += 1
        ## Update Vocabulary
        self._update_vocabulary()
        return self

    def transform(self,
                  X):
        """
        Isolate the top-k features based on pointwise
        mutual information

        Args:
            X (2d-array): Feature matrix of original size used
                          for learning pmi
        
        Returns:
            X_red (2d-array): Document-term matrix (reduced size)
        """
        X_red = X[:, self.top_k]
        return X_red

    def fit_transform(self,
                      X,
                      y):
        """
        Apply fit and transform methods in sequence

        Args:
            X (2d-array): Feature Matrix (Count-based)
            y (1d-array): Class Labels
        
        Returns:
            X_T (2d-array): Reduced feature set
        """
        ## Call Fit Method
        _ = self.fit(X, y)
        ## Call Transform Method
        X_T = self.transform(X)
        ## Return
        return X_T

    def get_pmi(self):
        """
        Return a dataframe containing PMI values
        for each class, along with baseline feature
        frequency

        Args:
            None

        Returns:
            df (pandas DataFrame): PMI Information for Each Feature
        """
        ## Initialize Score DataFrame
        df = pd.DataFrame(data=self.pmi,
                          index=self._classes,
                          columns=self._original_vocab).T
        ## Add Frequency Statistics
        df["frequency"] = self.freq
        if self._binarize:
            df.rename(columns={"frequency":"user_frequency"},
                      inplace=True)
        return df

##################
### Feature Selector Class
##################

class FeatureSelector(object):

    """
    Feature Selector
    """

    def __init__(self,
                 vocab,
                 selector=None,
                 feature_selection_kwargs={}):
        """
        Wrapper for feature selection.

        Args:
            vocab (Vocabulary): Initial vocabulary object
            selector (str or None): Name of the feature selector to use
                                    or None if no feature-selection desired.
            feature_selection_kwargs (dict): Arguments to pass to initialization
                                             of the chosen feature selector.
        """
        ## Class Attributes
        self.vocab = vocab
        self._selector = selector
        self._feature_selection_kwargs = feature_selection_kwargs
        ## Initialize Selector
        self._initialize_selector()
    
    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        desc = f"FeatureSelector(selector={self._selector})"
        return desc
    
    def _initialize_selector(self):
        """
        Initialize the feature selector class.

        Args:
            None
        
        Returns:
            None, sets the self._selector class attribute in place.
        """
        if not hasattr(self, "_selector") or self._selector is None:
            self._selector = None
            return
        elif self._selector == "kldivergence":
            self._selector = KLDivergenceSelector(vocab=self.vocab,
                                                  **self._feature_selection_kwargs)
        elif self._selector == "pmi":
            self._selector = PMISelector(vocab=self.vocab,
                                         **self._feature_selection_kwargs)
    
    def update_model_vocabulary(self,
                                model):
        """
        Update a classifier's vocabulary using a reduced vocabulary.

        Args:
            model (Classifier class): e.g. MentalHealthClasifier
        
        Returns:
            model (Classifier class): input model with newly aligned
                                      vocabulary and vectorization class.
        """
        ## Check to see if there is any feature selector in place
        if not hasattr(self, "_selector") or self._selector is None:
            return model
        ## Update model vocab and vectorization class
        model.vocab = self._selector.vocab
        _ = model._initialize_dict_vectorizer()
        return model

    def fit(self,
            X,
            y=None):
        """
        Fit the chosen feature selector.

        Args:
            X (2d-array): Input feature matrix to be passed to feature
                          selector
            y (1d-array or None): If feature selector requires labels,
                                  pass here as a vector.
        
        Returns:
            self
        """
        ## Check to see if there is any feature selector in place
        if not hasattr(self, "_selector") or self._selector is None:
            return self
        ## Fit the Feature Selector
        self = self._selector.fit(X, y)
        return self
    
    def transform(self,
                  X):
        """
        Using a fit feature selector, apply feature selection to a feature matrix.

        Args:
            X (2d-array): Feature matrix. Dimensions should correspond with those
                          used to fit the feature selector.
        
        Returns:
            X_red (2d-array): Reduced dimensionality represention of X.
        """
        ## Check to see if there is any feature selector in place
        if not hasattr(self, "_selector") or self._selector is None:
            return X
        ## Perform Dimensionality Reduction
        logger.info("Reducing Vocabulary Dimensionality. Currently = {}".format(X.shape[1]))
        X_red = self._selector.transform(X)
        logger.info("New Dimensionality = {}".format(X_red.shape[1]))
        return X_red
    
    def fit_transform(self,
                      X,
                      y=None):
        """
        Fit the class feature selector and retrieve the reduced representation
        for the input feature matrix X.

        Args:
            X (2d-array): Input feature matrix to be passed to feature
                          selector
            y (1d-array or None): If feature selector requires labels,
                                  pass here as a vector.
        
        Returns:
            X_red (2d-array): Reduced dimensionality represention of X.            
        """
        ## Fit the Selector
        self.fit(X, y)
        ## Reduce Dimensionality of the Dataset
        X_red = self.transform(X)
        return X_red

