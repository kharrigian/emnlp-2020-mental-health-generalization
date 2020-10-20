
##################
### Imports
##################

## Standard Library
import os
import json
from copy import deepcopy
from itertools import product
from pprint import PrettyPrinter
import uuid

## External Libraries
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

## Local Modules
from .classifiers import *
from .feature_selectors import FeatureSelector
from .feature_extractors import FeaturePreprocessor
from ..util.multiprocessing import MyPool as Pool
from ..util.helpers import flatten
from ..util.logging import initialize_logger

##################
### Globals
##################

## LOGGER
LOGGER = initialize_logger()

## Pretty Printer
pp = PrettyPrinter()

## Temp Directory For Storing Data and Objects
ROOT_DIR =  os.path.abspath(os.path.dirname(__file__)) + "/../../"
TEMP_DIR_PREFIX = f"{ROOT_DIR}grid_search_temp_"

##################
### Base Helpers
##################

def _dict_product(dicts):
    """
    Get all unique dictionaries created as a function of the lists
    they contain as valus.

    Args:
        dicts (dict): {"key":[]} format
    
    Returns:
        >>> dict_product(dict(number=[1,2], character='ab'))
        [{'character': 'a', 'number': 1},
        {'character': 'a', 'number': 2},
        {'character': 'b', 'number': 1},
        {'character': 'b', 'number': 2}]
    """
    return list(dict(zip(dicts, x)) for x in product(*dicts.values()))

##################
### Grid Search Helpers
##################

def _log_result_dict(result_dict):
    """
    Use PrettyPrint to format and log a hyperparameter
    result dictionary

    Args:
        result_dict (dict): Any dictionary
    
    Returns:
        None, logs dictionary to standard output
    """
    rd_formatted = pp.pformat(result_dict)
    LOGGER.critical(rd_formatted)

def _check_nonablation_is_available(features):
    """

    """
    nonablation_valid = True
    for feature_type, feature_args in features.items():
        for arg, arg_vals in feature_args.items():
            if len(arg_vals) > 1:
                nonablation_valid = False
    return nonablation_valid

def _construct_grid_search_pipeline(config=None):
    """
    Build out a data object so that it's easy to construct
    a modeling grid search pipeline.

    Args:
        config (str or None): Path to hyperparameter search config. Default
                              is None and loads "_default" configuration.
    
    Returns:
        hsearch_space (list of dict): Processed configuration
    """
    ## Load Config
    if config is None:
        config = os.path.dirname(__file__) + "/../../configurations/hyperparameter_search/_default.json"
    if not os.path.exists(config):
        raise FileNotFoundError(f"Could not find hyperparameter config file: {config}")
    with open(config, "r") as the_file:
        config = json.load(the_file)
    ## Level 0: Sets of Hyperparameter Searches
    hsearch_space = []
    for hsearch in config:
        ## Level 1: Feature Selection
        feature_selectors = []
        if hsearch["feature_selector_ablation"]:
            feature_selectors.append((None, {}))
        for selector, selector_params in hsearch["feature_selector"].items():
            selector_grid = _dict_product(selector_params)
            feature_selectors.extend((selector, grid_set) for grid_set in selector_grid)
        ## Level 2: Feature Set Generation (Ablation, will use best previous feature setting)
        feature_sets = []
        ablation_order = ["bag_of_words","tfidf","liwc","lda","glove"]
        for a, added_feature in enumerate(ablation_order):
            if added_feature in hsearch["features"]:
                added_feature_params = hsearch["features"][added_feature]
                added_feature_grid = _dict_product(added_feature_params)
                if added_feature == "bag_of_words":
                    added_feature_sets = [{"bag_of_words":{}}]
                elif added_feature == "tfidf" and "bag_of_words" in hsearch["features"]:
                    added_feature_sets = []
                    for gset in added_feature_grid:
                        added_feature_sets.append({"tfidf":gset})
                else:
                    added_feature_sets = []
                    if len(feature_sets) > 0:
                        base_feature_set = dict((i, {}) for i in feature_sets[-1][0].keys())
                    else:
                        base_feature_set = dict()
                    for gset in added_feature_grid:
                        gset_copy = base_feature_set.copy()
                        gset_copy.update({added_feature:gset})
                        added_feature_sets.append(gset_copy)
                feature_sets.append(added_feature_sets)
        ## Drop Ablation, Use a Single Feature Set (If Called For)
        nonablation_valid = _check_nonablation_is_available(hsearch["features"])
        if not hsearch["feature_set_ablation"] and nonablation_valid:
            fset = dict(map(lambda f: (f[0], _dict_product(f[1])[0]) if len(f[1]) > 0 else f, hsearch["features"].items()))
            if "bag_of_words" in fset and "tfidf" in fset:
                _ = fset.pop("bag_of_words", None)
            feature_sets = [[fset]]
        elif not hsearch["feature_set_ablation"] and not nonablation_valid:
            LOGGER.critical("Attempted to skip feature ablation, but feature sets are non-unique.")
        ## Level 3: Additional Preprocessing (e.g Standardization)
        feature_sets_arguments = []
        for ablation_group in feature_sets:
            for standardize in hsearch["standardize"]:
                ablation_group_args = []
                for features in ablation_group:
                    feature_dict = {"feature_flags":dict((f, True) for f in features.keys()),
                                    "feature_kwargs":features.copy(),
                                    "standardize":standardize}
                    ablation_group_args.append(feature_dict)
                feature_sets_arguments.append(ablation_group_args)
        ## Level 4: Model Hyperparameters
        model_hyperparams = []
        for model_group in hsearch["model"]:
            model_name = model_group.pop("name", None)
            model_param_grid = _dict_product(model_group)
            for mg in model_param_grid:
                mg_args = {"model":model_name,
                           "model_kwargs":mg.copy()}
                model_hyperparams.append(mg_args)
        hsearch_space.append(
            {"feature_selectors":feature_selectors,
             "feature_sets":feature_sets_arguments,
             "model_hyperparameters":model_hyperparams})
    return hsearch_space

def _fit_model(params):
    """
    Fit a single mental-health classifier model and score its performance
    on the training/development data

    Args:
        params (dict): Parameters passed by _fit_models
    
    Returns:
        res (dict): Name of the model, Model Parameters, and 
                    Training/Development Performance
    """
    ## Load the Data
    X_train = joblib.load(params.get("X_train"))
    X_dev = joblib.load(params.get("X_dev"))
    y_train = joblib.load(params.get("y_train"))
    y_dev = joblib.load(params.get("y_dev"))
    ## Format kwargs
    model_kwargs = params.get("model_params").get("model_kwargs")
    if params.get("model_params").get("model") == "mlp" and "hidden_layers" in model_kwargs:
        if isinstance(model_kwargs["hidden_layers"],list):
            model_kwargs["hidden_layers"] = tuple(model_kwargs["hidden_layers"])
    if "class_weight" in model_kwargs and isinstance(model_kwargs.get("class_weight"),dict):
        cw = {0:model_kwargs["class_weight"]["0"],
              1:model_kwargs["class_weight"]["1"]}
        model_kwargs["class_weight"] = cw
    if "random_state" in model_kwargs and params.get("model_params").get("model") == "naive_bayes":
        _ = model_kwargs.pop("random_state", None)
    ## Drop Null
    if params.get("drop_null"):
        train_mask = np.nonzero((X_train!=0).any(axis=1))[0]
        dev_mask = np.nonzero((X_dev!=0).any(axis=1))[0]
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_dev = X_dev[dev_mask]
        y_dev = y_dev[dev_mask]
    ## Initialize Model
    classifier = MODEL_DICT.get(params.get("model_params").get("model"))(**model_kwargs)
    ## Fit Model
    classifier = classifier.fit(X_train,
                                y_train)
    ## Make Predictions
    y_train_preds = classifier.predict(X_train)
    y_dev_preds = classifier.predict(X_dev)
    ## Score Model Performance
    train_score = params.get("score_func")(y_train, y_train_preds)
    dev_score = params.get("score_func")(y_dev, y_dev_preds)
    ## Store Results
    res = {"name":params.get("model_params").get("model"),
           "model_kwargs":params.get("model_params").get("model_kwargs"),
           "train_score":train_score,
           "dev_score":dev_score}
    return res

def _construct_preprocessed_objects(vocab,
                                    X_train,
                                    X_dev,
                                    ag_params,
                                    temp_dir):
    """

    """
    ## Load the Data
    X_train = joblib.load(X_train)
    X_dev = joblib.load(X_dev)
    vocab = joblib.load(vocab)
    ag_params = ag_params.copy()
    ## Apply Preprocessing to Generate Features
    preprocessor = FeaturePreprocessor(vocab,
                                       **ag_params)
    X_train = preprocessor.fit_transform(X_train)
    X_dev = preprocessor.transform(X_dev)
    ## Cache Temporary Objects
    object_paths = {}
    for obj, obj_name in zip([X_train, X_dev],
                             ["X_train_preprocessed","X_dev_preprocessed"]):
        rand_id = str(uuid.uuid4())
        temp_obj_path = f"{temp_dir}{obj_name}_{rand_id}.joblib"
        _ = joblib.dump(obj, temp_obj_path)
        object_paths[obj_name] = temp_obj_path
    del X_train, X_dev, vocab
    return object_paths

def _fit_models(params):
    """
    Run a grid search over model hyperparameters

    Args:
        params (dict): Parameters passed by _get_ablation_results
    
    Returns:
        model_res_appended (list): All model hyperparameter results.
    """
    ## Copy AG Params
    ag_params = params.get("ag_params").copy()
    ## Construct Preprocessed Objects
    preprocessed_object_paths = _construct_preprocessed_objects(vocab=params.get("vocab"),
                                                                X_train=params.get("X_train"),
                                                                X_dev=params.get("X_dev"),
                                                                ag_params=params.get("ag_params"),
                                                                temp_dir=params.get("temp_dir"))
    ## Iterator for Model Fitting
    model_params = iter(map(lambda m: {
                    "model_params":m,
                    "X_train":preprocessed_object_paths["X_train_preprocessed"],
                    "X_dev":preprocessed_object_paths["X_dev_preprocessed"],
                    "y_train":params.get("y_train"),
                    "y_dev":params.get("y_dev"),
                    "score_func":params.get("score_func"),
                    "drop_null":params.get("drop_null")}, params.get("hspace")["model_hyperparameters"]))
    ## Multiprocessing of Model Hyperparameters
    if params.get("use_multiprocessing"):
        model_pool = Pool(params.get("jobs"))
        model_res = list(model_pool.map(_fit_model, model_params))
        model_pool.close()
    else:
        model_res = []
        for mp in model_params:
            model_res.append(_fit_model(mp))
    ## Append Feature Parameters
    model_res_appended = []
    for m in model_res:
        ## Add Feature Parameters to Model Results
        m["feature_params"] = ag_params.copy()
        ## Remove Vocab From Being Stored (Unneccessary)
        for f, f_dict in m["feature_params"]["feature_kwargs"].items():
            if "vocab" in f_dict:
                _ = f_dict.pop("vocab",None)
        ## Update Model Result Cache
        model_res_appended.append(m)
    ## Remove Temporary Objects
    for obj_name, obj_file in preprocessed_object_paths.items():
        _ = os.system(f"rm {obj_file}")
    return model_res_appended

def _get_ablation_results(vocab,
                          X_train,
                          y_train,
                          X_dev,
                          y_dev,
                          temp_dir,
                          num_jobs,
                          use_multiprocessing,
                          ablation_group_params,
                          hspace,
                          drop_null=False,
                          score_func=f1_score):
    """
    Run a grid search one set of features (an ablation grouping)

    Args:
        model (MentalHealthClassifer): Initialized mental-health classifier
        X_train (2d-array): Base Bag-of-words Training Matrix
        y_train (1d-array): Vectorized training labels
        X_dev (2d-array): Base Bag-of-words Development Matrix
        y_dev (1d-array): Vectorized development labels
        temp_dir (str):
        ablation_group_params (list): List of feature parameters to pass to FeaturePreprocessor
        hspace (dict): Hyperparameter search space configuration
        score_func (function): Score model performance. Should assume higher is better.
    
    Returns:
        ablation_res_flat (list): All ablation results.
        best_result (dict): Parameters that maximize development score
        best_feature_set_score (float): Maximal development score found during the grid search
    """
    ## Combine Params into Dict for Multiprocessing
    ablation_params = iter(map(lambda ag: {"X_train":X_train,
                                           "y_train":y_train,
                                           "X_dev":X_dev,
                                           "y_dev":y_dev,
                                           "temp_dir":temp_dir,
                                           "score_func":score_func,
                                           "vocab":vocab,
                                           "hspace":hspace,
                                           "jobs":num_jobs,
                                           "use_multiprocessing":use_multiprocessing,
                                           "ag_params":ag,
                                           "drop_null":drop_null}, ablation_group_params))
    ## Execute and Return
    if use_multiprocessing:
        ablation_pool = Pool(num_jobs)
        ablation_res = list(ablation_pool.map(_fit_models, ablation_params))
        ablation_pool.close()
    else:
        ablation_res = []
        for a_params in ablation_params:
            ablation_res.append(_fit_models(a_params))
    ## Flatten Ablation Results
    ablation_res_flat = flatten(ablation_res)
    ## Get Best Result and Best Score
    best_result = None
    best_feature_set_score = -np.inf
    for res_dict in ablation_res_flat:
        if res_dict["dev_score"] > best_feature_set_score:
            best_feature_set_score = res_dict["dev_score"]
            best_result = res_dict
    return ablation_res_flat, best_result, best_feature_set_score

def _construct_reduced_objects(vocab,
                               X_train,
                               y_train,
                               X_dev,
                               y_dev,
                               temp_dir,
                               feature_selector,
                               feature_selector_kwargs):
    """

    """
    ## Load Data
    X_train = joblib.load(X_train)
    y_train = joblib.load(y_train)
    X_dev = joblib.load(X_dev)
    vocab =  joblib.load(vocab)
    ## Initialize Feature Selector
    selector = FeatureSelector(vocab=vocab,
                               selector=feature_selector,
                               feature_selection_kwargs=feature_selector_kwargs)
    ## Fit and Transform Bag-of-words Vocabulary
    X_train = selector.fit_transform(X_train, y_train)
    X_dev = selector.transform(X_dev)
    ## Cache Temp Objects
    object_paths = {}
    for obj, obj_name in zip([X_train, X_dev, vocab],
                             ["X_train_reduced","X_dev_reduced","vocab_reduced"]):
        rand_id = str(uuid.uuid4())
        obj_temp_file = f"{temp_dir}{obj_name}_{rand_id}.joblib"
        _ = joblib.dump(obj, obj_temp_file)
        object_paths[obj_name] = obj_temp_file
    del X_train, X_dev, y_train, vocab
    return object_paths

def _run_hspace(X_train,
                y_train,
                X_dev,
                y_dev,
                vocab,
                temp_dir,
                num_jobs,
                use_multiprocessing=True,
                hspace=None,
                drop_null=False,
                score_func=f1_score):
    """
    Run a grid search over one of the hyperparameter search space groups

    Args:
        model (MentalHealthClassifier): Initialized mental-health classifier
        X_train (2d-array): Base Bag-of-words Training Matrix
        y_train (1d-array): Vectorized training labels
        X_dev (2d-array): Base Bag-of-words Development Matrix
        y_dev (1d-array): Vectorized development labels
        temp_dir (str):
        hspace (list)
        score_func (function): Score model performance. Should assume higher is better.
    
    Returns:
        hspace_results (list): All search results.
        hspace_best_result (dict): Parameters that maximize development score
        hspace_best_score (float): Maximal development score found during the grid search
    """
    ## Cache of Results
    hspace_results = []
    hspace_best_result = None
    hspace_best_score = -np.inf
    ## First Level Loop: Feature Selectors
    for feature_selector, feature_selector_kwargs in hspace.get("feature_selectors"):
        ## Apply Feature Selection
        reduced_object_paths = _construct_reduced_objects(vocab,
                                                          X_train,
                                                          y_train,
                                                          X_dev,
                                                          y_dev,
                                                          temp_dir,
                                                          feature_selector,
                                                          feature_selector_kwargs)
        ## Level 2: Feature Sets (Ablation)
        best_ablation_feature_set = None
        best_ablation_feature_set_score = -np.inf
        for ablation_group in hspace["feature_sets"]:
            ablation_group_params = []
            for ag in ablation_group:
                if best_ablation_feature_set is not None:
                    ## Update Group Params
                    ag_copy = ag.copy()
                    ## Check to See if We Are Replace BOW with TFIDF
                    if best_ablation_feature_set.get("feature_params")["feature_flags"] == \
                        {"bag_of_words":True} and "tfidf" in ag_copy["feature_flags"]:
                        pass
                    else:
                        ag_copy["feature_flags"].update(best_ablation_feature_set.get("feature_params").get("feature_flags"))
                        ag_copy["feature_kwargs"].update(best_ablation_feature_set.get("feature_params").get("feature_kwargs"))
                    ablation_group_params.append(ag_copy)
                else:
                    ablation_group_params.append(ag)
            ## Run Models on All Ablation Group Params
            ablation_results, best_result, best_score = _get_ablation_results(reduced_object_paths["vocab_reduced"],
                                                                  reduced_object_paths["X_train_reduced"],
                                                                  y_train,
                                                                  reduced_object_paths["X_dev_reduced"],
                                                                  y_dev,
                                                                  temp_dir,
                                                                  num_jobs,
                                                                  use_multiprocessing,
                                                                  ablation_group_params,
                                                                  hspace,
                                                                  drop_null=drop_null,
                                                                  score_func=score_func)
            ## Update Best Result Attributes
            best_result["feature_selector"] = feature_selector
            best_result["feature_selector_kwargs"] = feature_selector_kwargs
            ## Add Feature Selector Information to All Ablation Results
            ablation_results_appended = []
            for a in ablation_results:
                a_cop = a.copy()
                a_cop["feature_selector"] = feature_selector
                a_cop["feature_selector_kwargs"] = feature_selector_kwargs
                ablation_results_appended.append(a_cop)
            ## Cache Ablation Results
            hspace_results.extend(ablation_results_appended)
            ## Update Within-Feature Ablation Best Result (e.g. Do not add feature if it doesn't help)
            if best_score >= best_ablation_feature_set_score:
                best_ablation_feature_set = best_result.copy()
                best_ablation_feature_set_score = best_score
                ## Update Within-Hyperparameter Search Space Best Result
                if best_score >= hspace_best_score:
                    LOGGER.critical("~"*50)
                    LOGGER.critical("New Best Feature Set with Score of {:.5f}...".format(best_score))
                    _ = _log_result_dict(best_ablation_feature_set)
                    LOGGER.critical("~"*50)
                    hspace_best_score = best_score
                    hspace_best_result = best_ablation_feature_set.copy()
                    hspace_best_result["feature_selector"] = feature_selector
                    hspace_best_result["feature_selector_kwargs"] = feature_selector_kwargs
                else:
                    LOGGER.critical("."*10 + " Best Score Unchanged " + "."*10)
            else:
                LOGGER.critical("."*10 + " Best Score Unchanged " + "."*10)
        ## Remove Temporary Objects
        for obj_name, obj_file in reduced_object_paths.items():
            _ = os.system(f"rm {obj_file}")
    return hspace_results, hspace_best_result, hspace_best_score

def _construct_base_objects(model,
                            train_files,
                            train_label_dict,
                            dev_files,
                            dev_label_dict,
                            random_state,
                            test_size,
                            dev_in_vocab,
                            temp_dir):
    """

    """
    ## If No Development Files Supplied, Make Split
    if dev_files is None:
        train_files, dev_files = train_test_split(train_files,
                                                  test_size=test_size,
                                                  random_state=random_state)
        dev_label_dict = train_label_dict.copy()
    ## Learn Vocabulary
    vocab_learning_files = train_files if not dev_in_vocab else train_files + dev_files
    _ = model._learn_vocabulary(vocab_learning_files)
    ## Load Vectors
    train_files, X_train, y_train = model._load_vectors(train_files,
                                                       train_label_dict,
                                                       min_date=model._min_date,
                                                       max_date=model._max_date,
                                                       n_samples=model._n_samples,
                                                       randomized=model._randomized)
    dev_files, X_dev, y_dev = model._load_vectors(dev_files,
                                                  dev_label_dict,
                                                  min_date=model._min_date,
                                                  max_date=model._max_date,
                                                  n_samples=model._n_samples,
                                                  randomized=model._randomized)
    ## Cache Data Objects
    object_paths = {}
    for obj, obj_name in zip([X_train, y_train, X_dev, y_dev, model.vocab],
                             ["X_train_base","y_train_base","X_dev_base","y_dev_base","vocab_base"]):
        rand_id = str(uuid.uuid4())
        base_temp_file = f"{temp_dir}{obj_name}_{rand_id}.joblib"
        _ = joblib.dump(obj, base_temp_file)
        object_paths[obj_name] = base_temp_file
    ## Clear Objects from Memory
    del X_train, X_dev, y_train, y_dev
    return object_paths

##################
### Primary Function
##################
         
def run_grid_search(model,
                    train_files,
                    train_label_dict,
                    dev_files=None,
                    dev_label_dict=None,
                    config=None,
                    random_state=42,
                    test_size=.2,
                    dev_in_vocab=False,
                    use_multiprocessing=True,
                    drop_null=False,
                    score_func=f1_score):
    """
    Run a grid search over model and feature hyperparameters to optimize generalization
    performance. Uses multiprocessing where possible to sped up search, but may still 
    take long for larger search spaces.

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
        model (MentalHealthClassifier): Initialized mental-health classifier
        train_files (list): List of filenames for training the model
        train_label_dict (dict): Mapping between filenames and mental-health labels
        dev_files (list or None): List of filenames to use as development data. If None,
                                  we will automatically split the training files
        dev_label_dict (dict): Mapping between dev filenames and mental-health labels
        config (str or None): Path to hyperparameter optimization configuration. If None, uses
                              _default.json
        random_state (int): Random state for splitting data et al.
        test_size (float [0,1]): If dev files not supplied, how large of a sample to use for development
        dev_in_vocab (bool): If True, use the development data to construct the vocabulary
        score_func (function): Score model performance. Should assume higher is better.
    
    Returns:
        grid_search_results (list): All search results.
        grid_search_best_result (dict): Parameters that maximize development score
        grid_search_best_score (float): Maximal development score found during the grid search
    """
    ## Update LOGGER Level, Caching Original
    current_logger_level = LOGGER.level
    LOGGER.setLevel(50)
    ## Create Temporary Directory for Grid Search Data
    run_id = str(uuid.uuid4())
    temp_dir = f"{TEMP_DIR_PREFIX}{run_id}/"
    _ = os.makedirs(temp_dir)
    ## Get Hyperparameter Search Parameters
    hyperparameter_search_space = _construct_grid_search_pipeline(config=config)
    ## Create Base Objects
    base_object_paths = _construct_base_objects(model,
                                                train_files,
                                                train_label_dict,
                                                dev_files,
                                                dev_label_dict,
                                                random_state,
                                                test_size,
                                                dev_in_vocab,
                                                temp_dir)
    ## Cycle Through Hyperparameter Groups
    grid_search_results = []
    grid_search_best_result = None
    grid_search_best_score = -np.inf
    for hspace in hyperparameter_search_space:
        hspace_res, hspace_best_result, hspace_best_score = _run_hspace(vocab=base_object_paths["vocab_base"],
                                                                        X_train=base_object_paths["X_train_base"],
                                                                        y_train=base_object_paths["y_train_base"],
                                                                        X_dev=base_object_paths["X_dev_base"],
                                                                        y_dev=base_object_paths["y_dev_base"],
                                                                        temp_dir=temp_dir,
                                                                        num_jobs=model._jobs,
                                                                        use_multiprocessing=use_multiprocessing,
                                                                        hspace=hspace,
                                                                        drop_null=drop_null,
                                                                        score_func=score_func)
        grid_search_results.extend(hspace_res)
        if hspace_best_score > grid_search_best_score:
            grid_search_best_score = hspace_best_score
            grid_search_best_result = hspace_best_result
    ## Remove Temporary Objects and Directory
    for obj_name, obj_file in base_object_paths.items():
        _ = os.system(f"rm {obj_file}")
    _ = os.system(f"rm -rf {temp_dir}")
    ## Provide User with Result of the Search
    LOGGER.critical("~"*50)
    LOGGER.critical("Best Overall Score was {:.5f}...".format(hspace_best_score))
    LOGGER.critical("~"*50)
    ## Update LOGGER Level
    LOGGER.setLevel(current_logger_level)
    return grid_search_results, grid_search_best_result, grid_search_best_score
    