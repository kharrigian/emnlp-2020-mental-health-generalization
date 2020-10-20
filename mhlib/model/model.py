
##################
### Imports
##################

## Standard Library
import sys
from copy import deepcopy
from datetime import datetime

## External Libraries
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid

## Local Modules
from .classifiers import *
from .feature_selectors import FeatureSelector
from .feature_extractors import FeaturePreprocessor
from .file_vectorizer import File2Vec
from ..util.logging import initialize_logger
from .grid_search import run_grid_search

##################
### Globals
##################

## Create Logger
LOGGER = initialize_logger()

## Default Preprocessing Params (e.g. Feature Set)
DEFAULT_PREPROCESSING_KWARGS = {"feature_flags": {
                                                "bag_of_words":True
                                                 },
                                "feature_kwargs":{},
                                "standardize":True,
}

## Update Model Dict
CLASSIFIERS = {**MODEL_DICT}
CLASSIFIERS_PARAMS = {**DEFAULT_PARAMETERS}
CLASSIFIERS_PARAMETER_GRID = {**PARAMETER_GRID}

##################
### Helpers
##################

def _format_parameter_grid(grid):
    """

    """
    pgrid_formatted = {}
    for key, val in grid.items():
        if isinstance(val, list):
            pgrid_formatted[(key, )] = val
        elif isinstance(val, dict):
            for key_, val_ in val.items():
                if isinstance(val_, list):
                    pgrid_formatted[(key, key_)] = val_
                else:
                    raise ValueError("Too many levels in hyperparameter grid.")
        else:
            raise ValueError("Expected either a list or dictionary in value of hyperparameter grid (level 1)")
    return pgrid_formatted

##################
### Modeling Class
##################

class MentalHealthClassifier(File2Vec):

    """
    Mental Health Status Classification
    """

    def __init__(self,
                 target_disorder,
                 model="logistic",
                 model_kwargs={},
                 vocab_kwargs={},
                 preprocessing_kwargs=DEFAULT_PREPROCESSING_KWARGS,
                 feature_selector=None,
                 feature_selection_kwargs={},
                 min_date=None,
                 max_date=None,
                 n_samples=None,
                 randomized=False,
                 drop_null=False,
                 vocab_chunksize=10,
                 jobs=4,
                 random_state=42):
        """
        Mental Health Status Classifier wrapper.

        Args:
            target_disorder (str): Target mental health disorder (e.g. depression, ptsd)
            model (str): Classifier to use for modeling.
            model_kwargs (dict): Arguments to pass to initialization of the classifier.
            vocab_kwargs (dict): Arguments to pass to Vocabulary class
            preprocessing_kwargs (dict): Arguments to pass to preprocessor of document-
                                         term matrix.
            feature_selector (str or None): Name of the feature selector to use or None
                                            for performing dimensionality reduction.
            feature_selection_kwargs (dict): Arguments to pass to the feature selector
            min_date (str or None): ISO-format string representing lower bound in date range for
                                    training
            max_date (str or None): ISO-format string representing upper bound in date range for
                                    training 
            n_samples (int or None): Number of post samples to user for training. All if None.
            randomized (bool): If sampling posts, turning on this flag will sample randomly instead
                               of selecting the k most recent posts
            vocab_chunksize (int): Number of files to process during vocab learning 
                                   before adding to main counter object (memory preserving)
            jobs (int): Number of processes to use during the model learning procedure.
            random_state (int): Random state. Default is 42
        """
        ## Class Attributes
        self._target_disorder = target_disorder
        self._min_date = min_date
        self._max_date = max_date
        self._n_samples = n_samples
        self._randomized = randomized
        self._drop_null = drop_null
        self._jobs = jobs
        self._vocab_chunksize = vocab_chunksize
        self._random_state = random_state
        ## Initialize Class kwargs
        self._initialize_class_kwargs(vocab_kwargs,
                                      model,
                                      model_kwargs,
                                      feature_selector,
                                      feature_selection_kwargs,
                                      preprocessing_kwargs)
        ## Initialize Date Boundaries
        self._initialize_date_bounds()

    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        if not hasattr(self, "_vocab_kwargs") or len(self._vocab_kwargs) == 0:
            vs = ""
        else:
            vs = ", ".join("{}={}".format(x,y) for x,y in self._vocab_kwargs.items())
        if not hasattr(self, "_preprocessing_kwargs") or len(self._preprocessing_kwargs) == 0:
            ps = ""
        else:
            ps = ", ".join("{}={}".format(x,y) for x,y in self._preprocessing_kwargs.items())
        if not hasattr(self, "_target_disorder"):
            self._target_disorder = None
        desc = "MentalHealthClassifier(target_disorder={}, model={}, {}, {})".format(
            self._target_disorder if hasattr(self, "_target_disorder") else None,
            self.model,
            vs,
            ps)
        desc = desc.replace(", ,","").replace(" )",")")
        return desc
    
    def _initialize_class_kwargs(self,
                                 vocab_kwargs,
                                 model,
                                 model_kwargs,
                                 feature_selector,
                                 feature_selection_kwargs,
                                 preprocessing_kwargs):
        """
        Initialize class attributes, checking that random states
        are set uniformly across sub-classes.

        Args:
            vocab_kwargs (dict): Vocabulary parameters
            model (str): Name of the estimator to use
            model_kwargs (dict): Arguments to pass to estimator
            feature_selector (str or None): Name of the feature selector
                                           to use if desired.
            feature_selection_kwargs (dict): Arguments to pass to feature
                                             selector class
            preprocessing_kwargs (dict): Feature set arguments.
        
        Returns:
            None, sets class attributes in class
        """
        ## Cache kwargs
        self._model = model
        self._vocab_kwargs = vocab_kwargs
        self._model_kwargs = model_kwargs
        self._preprocessing_kwargs = preprocessing_kwargs
        self._feature_selection_kwargs = feature_selection_kwargs
        ## Initialize Classification Model
        if len(self._model_kwargs) == 0:
            self._model_kwargs = CLASSIFIERS_PARAMS.get(model)
        if "random_state" not in model_kwargs:
            self._model_kwargs["random_state"] = self._random_state
        else:
            self._model_kwargs["random_state"] = self._random_state
        if model == "mlp" and "hidden_layers" in model_kwargs:
            if isinstance(model_kwargs.get("hidden_layers"), list):
                self._model_kwargs["hidden_layers"] = tuple(self._model_kwargs["hidden_layers"])
        if "class_weight" in model_kwargs and isinstance(self._model_kwargs.get("class_weight"),dict):
            cw = {0:self._model_kwargs["class_weight"]["0"],
                  1:self._model_kwargs["class_weight"]["1"]}
            self._model_kwargs["class_weight"] = cw
        if model == "naive_bayes" and "random_state" in self._model_kwargs:
            _ = self._model_kwargs.pop("random_state", None)
        self.model = CLASSIFIERS.get(model)(**self._model_kwargs)
        ## Feature Selection (store name or None)
        self.selector = feature_selector
        ## Randomization in Vocabulary
        if "random_state" not in self._vocab_kwargs:
            self._vocab_kwargs["random_state"] = self._random_state
        ## File2Vec Inheritence Initialization
        super(MentalHealthClassifier, self).__init__(vocab_kwargs)

    def _initialize_date_bounds(self):
        """
        Initialize data boundaries as datetime objects

        Args:
            None
        
        Returns:
            None, updates _min_date and _max_date parameters
        """
        if not hasattr(self, "_min_date"):
            self._min_date = None
        if not hasattr(self, "_max_date"):
            self._max_date = None
        if self._min_date is not None:
            self._min_date = pd.to_datetime(self._min_date)
        if self._max_date is not None:
            self._max_date = pd.to_datetime(self._max_date)

    def _learn_vocabulary(self,
                          filenames):
        """
        Fit a Vocabulary class based on preprocessed user data
        files.

        Args:
            filenames (list of str): Path to files to use for constructing
                                     the vocabulary.
        
        Returns:
            None, sets self.vocab in place.
        """
        ## Learn Vocabulary
        LOGGER.info("Learning Vocabulary")
        self.vocab = self.vocab.fit(filenames,
                                    chunksize=self._vocab_chunksize if hasattr(self, "_vocab_chunksize") else 10,
                                    jobs=self._jobs if hasattr(self, "_jobs") else 1,
                                    min_date=self._min_date if hasattr(self, "_min_date") else None,
                                    max_date=self._max_date if hasattr(self, "_max_date") else None,
                                    n_samples=self._n_samples if hasattr(self, "_n_samples") else None,
                                    randomized=self._randomized if hasattr(self, "_randomized") else False)
        ## Initialize Dict Vectorizer
        _ = self._initialize_dict_vectorizer()
    
    def get_feature_names(self):
        """
        Extract model feature names

        Args:
            None
        
        Returns:
            features (list): Named list of features in the processed feature matrix.
        """
        features = []
        if not hasattr(self.preprocessor, "_transformer_names"):
            self.preprocessor._transformer_names = []
        for t in self.preprocessor._transformer_names:
            transformer = self.preprocessor._transformers[t]
            if t in ["bag_of_words","tfidf"]:
                tf = self.vocab.get_ordered_vocabulary()
            elif t == "glove":
                tf =  list(map(lambda i: f"GloVe_Dim_{i+1}", range(transformer.dim)))
            elif t == "liwc":
                tf = [f"LIWC={n}" for n in transformer.names]
            elif t == "lda":
                tf = [f"LDA_TOPIC_{i+1}" for i in range(transformer.n_components)] 
            features.extend(tf)
        return features

    def _load_vectors(self,
                      filenames,
                      label_dict=None,
                      min_date=None,
                      max_date=None,
                      n_samples=None,
                      randomized=False,
                      return_post_counts=False):
        """
        Load the raw document-term matrix and (optionally)
        associated label vector for a list of user data files.

        Args:
            filenames (list of str): Preprocessed user data files
            label_dict (None or dict): If desired, load user labels
                                       as a vector.
            min_date (str, datetime, or None): Lower date bound
            max_date (str, datetime, or None): Upper date bound
            n_samples (int or None): Possible number of samples
            randomized (bool): Whether to use random sample selection instead
                               of most recent
            return_post_counts (bool): If True, return the number of posts used
                                    to generate each feature vector
        
        Returns:
            filenames (list of str): List of filenames associated with 
                                     rows in the feature matrix.
            X (2d-array): Raw document-term matrix.
            y (1d-array or None): Target classes if label_dict passed.
        """
        ## Check Attributes
        if not hasattr(self, "_target_disorder"):
            raise AttributeError("Did not find _target_disorder attribute in model class.")
        ## Vectorize the data (count-based)
        result = self._vectorize_files(filenames,
                                       self._jobs,
                                       min_date=min_date,
                                       max_date=max_date,
                                       n_samples=n_samples,
                                       randomized=randomized,
                                       return_post_counts=return_post_counts)
        if not return_post_counts:
            filenames, X = result
        else:
            filenames, X, n_posts = result
        ## Return Data if No Label Dict (e.g. Test/Dev setting)
        if label_dict is None:
            if not return_post_counts:
                return filenames, X, None
            else:
                return filenames, X, None, n_posts
        ## Format Labels
        LOGGER.info("Formatting Labels")
        y = self._vectorize_labels(filenames,
                                   label_dict,
                                   pos_class=self._target_disorder)
        if y.shape[0] != X.shape[0]:
            raise Exception("Learned label length not equal to feature matrix shape")
        if not return_post_counts:
            return filenames, X, y
        else:
            return filenames, X, y, n_posts
    
    def fit_with_grid_search(self,
                             train_files,
                             train_label_dict,
                             dev_files=None,
                             dev_label_dict=None,
                             config=None,
                             test_size=.2,
                             dev_in_vocab=False,
                             use_multiprocessing=True,
                             score_func=f1_score,
                             cache_results=True,
                             return_training_preds=True):
        """
        Run a grid search over model and feature hyperparameters to optimize generalization
        performance. Uses multiprocessing where possible to sped up search, but may still 
        take long for larger search spaces. Uses best result from grid search to fit the classifier.

        Process is as follows:
        1. Perform Feature Selection (e.g. KL-Divergence Selector) as specified by one set
        of feature selection hyperparameters
        2. For each feature class (e.g. BOW, LIWC) in hyperparameter config:
            Add feature class to previous best feature set
            For this feature class, consider all feature hyperparameter combinations (e.g. GloVe dimension, normalization)
            For each feature hyperparameter combination:
                Train a classifier with all model hyperparameter combinations
            Select the best result (model + feature hyperparameter combination).
            Cache the current best feature hyperparameter combination to use for adding new features
        3. Cache the best result and repeat for each feature selection hyperparameter selection
        4. Return the overall best set of feature selection, feature set, and model hyperparameters

        Args:
            train_files (list): List of filenames for training the model
            train_label_dict (dict): Mapping between filenames and mental-health status labels
            dev_files (list or None): List of filenames to use as development data. If None,
                                    we will automatically split the training files
            dev_label_dict (dict): Mapping between dev filenames and mental-health status labels
            config (str or None): Path to hyperparameter optimization configuration. If None, uses
                                _default.json
            test_size (float [0,1]): If dev files not supplied, how large of a sample to use for development
            dev_in_vocab (bool): If True, use the development data to construct the vocabulary
            score_func (function): Score model performance. Should assume higher is better.
            cache_results (bool): If True, store results as an attribute of the class. Otherwise, leave
                                  within the function.
            return_training_preds (bool): If True, return training predictions in addition
                    to the model class itself. Useful for saving vectorization time.
        
        Returns:
            self: Base class with trained classifier.
            y_pred (dict): Training predictions, mapping between filename and model
                           prediction. Prediction is either a class (1 = target mental health class)
                           or positive mental health status probability based on availability of probs
                           in the base classifier.
        """
        ## Run Grid Search
        LOGGER.info(f"Starting Grid Search Using Config: {config}")
        gs_res, gs_best_res, gs_best_score = run_grid_search(model=self,
                                                             train_files=train_files,
                                                             train_label_dict=train_label_dict,
                                                             dev_files=dev_files,
                                                             dev_label_dict=dev_label_dict,
                                                             config=config,
                                                             random_state=self._random_state if hasattr(self, "_random_state") else 42,
                                                             test_size=test_size,
                                                             use_multiprocessing=use_multiprocessing,
                                                             dev_in_vocab=dev_in_vocab,
                                                             drop_null=self._drop_null,
                                                             score_func=score_func)
        ## Cache Results
        self._grid_search_results = None
        self._grid_search_best_result = None
        self._grid_search_best_score = None
        if cache_results:
            self._grid_search_results = gs_res
            self._grid_search_best_result = gs_best_res
            self._grid_search_best_score = gs_best_score
        ## Re-initialize Model Parameters
        LOGGER.info("Re-initializing Model with Optimal Parameters from Grid Search")
        self._initialize_class_kwargs(vocab_kwargs=self._vocab_kwargs,
                                      model=gs_best_res["name"],
                                      model_kwargs=gs_best_res["model_kwargs"],
                                      feature_selector=gs_best_res["feature_selector"],
                                      feature_selection_kwargs=gs_best_res["feature_selector_kwargs"],
                                      preprocessing_kwargs=gs_best_res["feature_params"])
        ## Fit Model
        LOGGER.info("Starting Standard Fit Procedure")
        return self.fit(train_files=train_files,
                        label_dict=train_label_dict,
                        return_training_preds=return_training_preds)

    def fit(self,
            train_files,
            label_dict,
            return_training_preds=True):
        """
        Args:
            train_files (list of str): Paths to the training files
            label_dict (dict): Map between filename and training label
            return_training_preds (bool): If True, return training predictions in addition
                    to the model class itself. Useful for saving vectorization time.
        
        Returns:
            self: Base class with trained classifier.
            y_pred (dict): Training predictions, mapping between filename and model
                           prediction. Prediction is either a class (1 = target mental health status)
                           or positive mental health status probability based on availability of probs
                           in the base classifier.
        """
        ## Learn Vocabulary in Training Files
        _ = self._learn_vocabulary(train_files)
        ## Load Vectors From Disk
        LOGGER.info("Vectorizing Training Data")
        train_files, X_train, y_train = self._load_vectors(train_files,
                                                           label_dict,
                                                           min_date=self._min_date if hasattr(self, "_min_date") else None,
                                                           max_date=self._max_date if hasattr(self, "_max_date") else None,
                                                           n_samples=self._n_samples if hasattr(self, "_n_samples") else None,
                                                           randomized=self._randomized if hasattr(self, "_randomized") else False)
        ## Feature Selection (Handles no-selector as well)
        if not hasattr(self, "vocab") or self.vocab is None:
            raise AttributeError("vocab attribute is missing from model class")
        if not hasattr(self, "selector"):
            raise AttributeError("selector attribute is missing from model class")
        self.selector = FeatureSelector(vocab=self.vocab,
                                        selector=self.selector,
                                        feature_selection_kwargs=self._feature_selection_kwargs if hasattr(self, "_feature_selection_kwargs") else {})
        X_train = self.selector.fit_transform(X_train,
                                              y_train)
        self = self.selector.update_model_vocabulary(self)
        ## Alert User to Null Feature Sets
        null_rows = (X_train==0).all(axis=1)
        LOGGER.info("Found {}/{} Null Training Rows ({} Control, {} Target)".format(
                    null_rows.sum(),
                    len(null_rows),
                    null_rows.sum()-y_train[null_rows].sum(),
                    y_train[null_rows].sum()
        ))
        ## Drop Null Rows During Training if Desired
        if hasattr(self, "_drop_null") and self._drop_null:
            LOGGER.info("Dropping null data points")
            mask = np.nonzero(null_rows==0)[0]
            X_train = X_train[mask]
            y_train = y_train[mask]
            train_files = [train_files[i] for i in mask]
        ## Preprocessing
        LOGGER.info("Generating Feature Set")
        self.preprocessor = FeaturePreprocessor(vocab=self.vocab,
                                                feature_flags=self._preprocessing_kwargs.get("feature_flags") if hasattr(self, "_preprocessing_kwargs") else {},
                                                feature_kwargs=self._preprocessing_kwargs.get("feature_kwargs") if hasattr(self, "_preprocessing_kwargs") else {},
                                                standardize=self._preprocessing_kwargs.get("standardize") if hasattr(self, "_preprocessing_kwargs") else False)
        X_train = self.preprocessor.fit_transform(X_train)
        ## Fit Model
        LOGGER.info("Fitting Classifier")
        self.model.fit(X_train, y_train)
        ## Return Training Predictions if Desired
        if return_training_preds:
            LOGGER.info("Making predictions on training data")
            try:
                y_pred = self.model.predict_proba(X_train)[:, 1]
            except:
                y_pred = list(map(int, self.model.predict(X_train)))
            return self, dict((x, y) for x,y in zip(train_files, y_pred))
        return self
    
    def predict(self,
                test_files,
                min_date=None,
                max_date=None,
                n_samples=None,
                randomized=False,
                drop_null=False):
        """
        Make mental health status predictions on a list of test user files.

        Args:
            test_files (list of str): Preprocessed user data files.
            min_date (str, datetime, or None): Lower date bound
            max_date (str, datetime, or None): Upper date bound
            n_samples (int or None): Number of post-level samples to consider
            randomized (bool): If True, sample randomly instead of most recent
            drop_null (bool): If True, do not make predictions for rows without any matched vocab
        
        Returns:
            y_pred (dict): Mental health status probability or class prediction
                           depending on base classifier availability. Maps
                           between filename and prediction.
        """
        ## Date Boundaries
        if min_date is not None and isinstance(min_date,str):
            min_date=pd.to_datetime(min_date)
        if max_date is not None and isinstance(max_date,str):
            max_date=pd.to_datetime(max_date)
        ## Vectorize the data
        LOGGER.info("Vectorizing Files")
        test_files, X_test, _ = self._load_vectors(test_files,
                                                   None,
                                                   min_date=min_date,
                                                   max_date=max_date,
                                                   n_samples=n_samples,
                                                   randomized=randomized)
        ## Alert User To Null Rows
        null_rows = (X_test==0).all(axis=1)
        LOGGER.info("Found {}/{} Null Rows".format(
                    null_rows.sum(),
                    len(null_rows),
        ))
        ## Drop Null Rows if Desired
        if drop_null:
            LOGGER.info("Dropping null data points")
            mask = np.nonzero(null_rows==0)[0]
            X_test = X_test[mask]
            test_files = [test_files[i] for i in mask]
        ## Apply Any Additional Preprocessing
        LOGGER.info("Generating Feature Set")
        X_test = self.preprocessor.transform(X_test)
        ## Apply Model
        LOGGER.info("Making Predictions")
        try:
            y_pred = self.model.predict_proba(X_test)[:, 1]
        except:
            y_pred = list(map(int, self.model.predict(X_test)))
        return dict((x, y) for x,y in zip(test_files, y_pred))

    def copy(self):
        """
        Make a copy of the MentalHealthClassifier class.

        Args:
            None
        
        Returns:
            deepcopy of the MentalHealthClassifier
        """
        return deepcopy(self)
    
    def dump(self,
             filename,
             compress=5):
        """
        Save the model instance to disk using joblib.

        Args:
            filename (str): Name of the model for saving.
            compress (int): Level of compression to pass to
                            joblib dump function.
        
        Returns:
            None, saves model to disk.
        """
        if not filename.endswith(".joblib"):
            filename = filename + ".joblib"
        _ = joblib.dump(self,
                        filename,
                        compress=compress)
        return