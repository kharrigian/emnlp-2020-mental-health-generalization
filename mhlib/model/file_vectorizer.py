

##################
### Imports
##################

## Standard Library
import os
import sys
from functools import partial

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import (csr_matrix, vstack)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

## Local Modules
from .vocab import Vocabulary
from ..util.multiprocessing import MyPool as Pool

##################
### Vectorization
##################

class File2Vec(object):
    
    """
    Preprocessed Data to Vectorization
    """

    def __init__(self,
                 vocab_kwargs={},
                 favor_dense=True):
        """
        File2Vec transforms preprocessed (e.g. tokenized) data files into a clean
        vector format that can be used for learning.

        Args:
            vocab_kwargs (dict): Arguments to pass to vocabulary class
            favor_dense (bool): If True, vectorization results in dense array instead 
                                of csr sparse matrix. Useful because certain preprocessing
                                steps (e.g. Standardization) and classifiers require 
                                dense data representations for learning.
        """
        ## Cache kwargs
        self._vocab_kwargs = vocab_kwargs
        ## Other Class Attributes
        self._favor_dense = favor_dense
        ## Initialize Vocabulary Class
        self.vocab = Vocabulary(**vocab_kwargs)
        ## Workspace Variables
        self._count2vec = None
    
    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        desc = f"File2Vec(vocab_kwargs={self._vocab_kwargs}, favor_dense={self._favor_dense})"
        return desc
    
    def _initialize_dict_vectorizer(self):
        """
        Initialize a vectorizer that transforms a counter dictionary
        into a sparse vector of counts (with a uniform feature index)

        Args:
            None
        
        Returns:
            None, initializes self._count2vec inplace.
        """
        self._count2vec = DictVectorizer(separator=":")
        self._count2vec.vocabulary_ = self.vocab.ngram_to_idx.copy()
        rev_dict = dict((y, x) for x, y in self.vocab.ngram_to_idx.items())
        self._count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
        return

    def _vectorize_labels(self,
                          keys,
                          label_dict,
                          pos_class="depression"):
        """
        Create a 1d-array with class labels for the model
        based on filenames and a label dictionary.

        Args:
            keys (list): Ordered list of keys for label_dict
            label_dict (dict): Map between keys and label
            pos_class (str): Positive Class Name. Default is "depression". Users with "control" will 
                             be labeled as 0. Users without either will be labeled as -1
        
        Returns:
            labels (1d-array): Vector of class targets.
        """
        labels = np.array(list(map(lambda f: 1 if label_dict[f]==pos_class else 0 if label_dict[f]=="control" else -1, keys)))
        labels = labels.astype(int)
        return labels

    def _vectorize_single_file(self,
                               filename,
                               min_date=None,
                               max_date=None,
                               n_samples=None,
                               randomized=False,
                               return_post_counts=False):
        """
        Vectorize the tokens in a preprocessed data file.

        Args:
            filename (str): Path to preprocessed text data
            min_date (str): Lower data boundary. ISO-format.
            max_date (str): Upper date boundary. ISO-format
            n_samples (int or None): Number of post samples
            randomized (bool): If True and n_samples is not None,
                               sample randomly. Otherwise, samples
                               most recent posts
            return_post_counts (bool): If True, return the number of posts used
                                       to generate the vector as well
        
        Returns:
            filename (str): Input filename
            vec (sparse vector): Vector of feature counts
        """
        ## Load File Data, Count Tokens
        token_counts = self.vocab._load_and_count(filename,
                                                  min_date=min_date,
                                                  max_date=max_date,
                                                  n_samples=n_samples,
                                                  randomized=randomized,
                                                  return_post_counts=return_post_counts)
        if return_post_counts:
            token_counts, n_posts = token_counts
        ## Vectorize
        vec = self._count2vec.transform(token_counts)
        if not return_post_counts:
            return (filename, vec)
        else:
            return (filename, vec, n_posts)

    def _vectorize_files(self,
                         filenames,
                         jobs=4,
                         min_date=None,
                         max_date=None,
                         n_samples=None,
                         randomized=False,
                         return_post_counts=False):
        """
        Vectorize several files tokens using multiprocessing.

        Args:
            filenames (list of str): Preprocessed text files
            jobs (int): Number of processes to use for vectorization
            min_date (str, datetime, or None): Date Lower Bound
            max_date (str, datetime, or None): Date Upper Bound
            n_samples (int or None): Number of post samples
            randomized (bool): If True and n_samples is not None,
                               sample randomly. Otherwise, samples
                               most recent posts
            return_post_counts (bool): If True, returns the number of individual
                                posts used to generate each feature vector.

        Returns:
            filenames (list): List of filenames (in case order changed during
                              multiprocessing)
            vectors (array): Sparse or dense document-term matrix (based on
                             class intiialization parameters)
            n_posts (array, optional): Array of post counts for each filename
        """
        ## Date Boundaries
        if min_date is not None and isinstance(min_date,str):
            min_date = pd.to_datetime(min_date)
        if max_date is not None and isinstance(max_date,str):
            max_date = pd.to_datetime(max_date)
        ## Get Vectors
        vectorizer = partial(self._vectorize_single_file,
                             min_date=min_date,
                             max_date=max_date,
                             n_samples=n_samples,
                             randomized=randomized,
                             return_post_counts=return_post_counts)
        mp_pool = Pool(processes=jobs)
        vectors = list(tqdm(mp_pool.imap_unordered(vectorizer,
                                                   filenames,
                                                   chunksize=min(100, len(filenames))),
                            total=len(filenames),
                            desc="Filecount",
                            file=sys.stdout))
        mp_pool.close()
        ## Clean Pool Result
        filenames = [v[0] for v in vectors]
        if return_post_counts:
            n_posts = np.array([v[2] for v in vectors])
        vectors = vstack(v[1] for v in vectors)
        ## Transform Into Dense
        if hasattr(self, "_favor_dense") and self._favor_dense and isinstance(vectors, csr_matrix):
            vectors = vectors.toarray()
        if not return_post_counts:
            return filenames, vectors
        else:
            return filenames, vectors, n_posts
