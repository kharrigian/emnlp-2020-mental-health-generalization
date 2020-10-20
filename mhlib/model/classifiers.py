
#######################
### Imports
#######################

## External Libraries
from sklearn import (linear_model,
                     svm,
                     ensemble,
                     tree,
                     naive_bayes,
                     neural_network)

#######################
### Globals
#######################

## General Random State
RANDOM_STATE = 42

## Model Classes
MODEL_DICT = {
    "logistic":linear_model.LogisticRegression,
    "svm":svm.SVC,
    "linear_svm":svm.LinearSVC,
    "perceptron":linear_model.Perceptron,
    "naive_bayes":naive_bayes.MultinomialNB,
    "decision_tree":tree.DecisionTreeClassifier,
    "random_forest":ensemble.RandomForestClassifier,
    "adaboost":ensemble.AdaBoostClassifier,
    "gradient_boosting":ensemble.GradientBoostingClassifier,
    "mlp":neural_network.MLPClassifier,
    "sgd":linear_model.SGDClassifier
}

## Probability-Aware Models
PROBABILITY_AWARE = set(["logistic","svm","naive_bayes","sgd"])

## Default Parameters
DEFAULT_PARAMETERS = {
    "logistic":{
                "C":1,
                "random_state":RANDOM_STATE,
                "max_iter":10000,
                "solver":"lbfgs"
                },
    "svm":{
            "C":1,
            "kernel":"rbf",
            "gamma":0.01,
            "shrinking":True,
            "probability":True,
            "max_iter":10000,
            "random_state":RANDOM_STATE
    },
    "linear_svm":{
                "penalty":"l2",
                "loss":"squared_hinge",
                "dual":True,
                "C":1,
                "random_state":RANDOM_STATE,
                "max_iter":10000
    },
    "perceptron":{
                "penalty":"l2",
                "alpha":0.01,
                "fit_intercept":True,
                "max_iter":10000,
                "shuffle":True,
                "eta0":1,
                "random_state":RANDOM_STATE,
                "early_stopping":True,
                "validation_fraction":0.1,
    },
    "naive_bayes":{
                "alpha":1,
                "fit_prior":True,
                "class_prior":None
    },
    "decision_tree":{
                "criterion":"gini",
                "splitter":"best",
                "max_depth":10,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "random_state":RANDOM_STATE,
                "presort":False,
    },
    "random_forest":{
                "n_estimators":100,
                "criterion":"gini",
                "max_depth":5,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "bootstrap":True,
                "oob_score":True,
                "random_state":RANDOM_STATE,
    },
    "adaboost":{
                "base_estimator":MODEL_DICT.get("decision_tree")(**{
                                "criterion":"gini",
                                "splitter":"best",
                                "max_depth":10,
                                "min_samples_split":2,
                                "min_samples_leaf":1,
                                "random_state":RANDOM_STATE,
                                "presort":False}),
                "learning_rate":1.0,
                "n_estimators":100,
                "random_state":RANDOM_STATE,
    },
    "gradient_boosting":{
        "loss":"deviance",
        "learning_rate":0.1,
        "n_estimators":100,
        "subsample":1.0,
        "criterion":"friendman_mse",
        "min_samples_split":2,
        "min_samples_leaf":1,
        "max_depth":3,
        "random_state":RANDOM_STATE
    }
}

## Parameter Grids
PARAMETER_GRID = {
    "logistic":{
        "C":[1e-2,1e-1,1e0,1e1,1e2],
        "random_state":[RANDOM_STATE],
        "penalty":["l2"],
        "solver":["lbfgs"],
        "n_jobs":[1],
        "max_iter":[10000],
    },
    "svm":{
        "C":[1e-2,1e-1,1e0,1e1,1e2],
        "kernel":["linear"],
        "probability":[True],
        "random_state":[RANDOM_STATE],
    },
    "linear_svm":{
        "C":[1e-2,1e-1,1e0,1e1,1e2],
        "penalty":["l1","l2","none"],
        "random_state":[RANDOM_STATE]
    },
    "perceptron":{
        "penalty":["l1","l2","none"],
        "alpha":[1e-2,1e-1,1e0,1e1,1e2],
        "shuffle":[True],
        "n_jobs":[1],
        "random_state":[RANDOM_STATE],
    },
    "naive_bayes":{
        "alpha":[1e-2,1e-1,1e0,1e1,1e2],
        "fit_prior":[True],
    },
    "decision_tree":{
        "criterion":["gini","entropy"],
        "random_state":[RANDOM_STATE],
        "max_depth":[2, 5, 10, None],
        "max_features":[None, "sqrt"],
        "presort":[True],
    },
    "random_forest":{
        "n_estimators":[5, 10, 25, 50, 100],
        "criterion":["gini","entropy"],
        "max_depth":[2, 5, 10, None],
        "max_features":[None, "sqrt"],
        "n_jobs":[1],
        "random_state":[RANDOM_STATE]
    },
    "adaboost":{
        "base_estimator":[tree.DecisionTreeClassifier],
        "n_estimators":[5, 10, 25, 50, 100],
        "learning_rate":[1e-3,1e-2,1e-1,1,10,100],
        "random_state":[RANDOM_STATE]
    },
    "gradient_boosting":{
        "loss":["deviance"],
        "learning_rate":[1e-3,1e-2,1e-1,1,10,100],
        "n_estimators":[5, 10, 25, 50, 100],
        "criterion":["friedman_mse"],
        "max_depth":[2, 5, 10, None],
        "random_state":[RANDOM_STATE]
    }
}
