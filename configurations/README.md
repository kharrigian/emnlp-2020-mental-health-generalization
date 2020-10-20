# Configurations

We use JSON configuration files to set up experiments, hyperparameter searches, and model training processes. Parameters are mostly the same in meaning across configuration files, so we will only describe them once in the first example set and add notes for new parameters found in other configuration templates.

### Experiments

These configuration files are used for scripts in `scripts/experiment/`.

**Domain Transfer**

An example template is found below with comments of what each field specifies.

```
{
    "experiment_name": "balanced_downsampled", // Name of the experiment, will be appended to a runtime.
    "train_data": "clpsych", // Which data set should be used for training the model
    "test_data": "clpsych", // Which data set you want to apply the model to (e.g. the domain of transfer)
    "target_disorder":"depression", // Which disorder is being modeled (e.g. depression, ptsd)
    "run_on_test": false, // Not currently supported. In the future, will apply trained models to test split after cross validation
    "jobs": 8, // Number of cores to use
    "random_seed": 42, // Random seed for the experiment splits and model.
    "kfolds": 5, // Number of cross-validation folds
    "stratified": true, // Whether cross-validation folds should use stratified splitting
    "test_size": 0.2, // How large of a held-out test set should be set aside
    "downsample":{ // Whether or not the data sets should be downsampled
        "train":true,
        "test":true
    },
    "downsample_size":{ // If downsampling turned on, how many examples (users) do you want in each data set
        "train":100,
        "test":100
    },
    "rebalance":{ // Whether or not to rebalance the class distribution before training
        "train":true,
        "test":true
    },
    "class_ratio": { // If rebalancing turned on, this controls the ratio between [target disorder to control]
                    "train": [1, 1],
                    "test": [1, 1]
    },
    "vocab_kwargs": { // Post-processing for vocabulary selection
        "filter_negate": true, // If turned on, filter any <NEGATE> tokens
        "filter_upper": true,  // If turned on, filter and <UPPER> tokens
        "filter_punctuation": true, // If turned on, filter out most punctuation
        "filter_numeric": true, // If turned on, filter out <NUMERIC> tokens
        "filter_user_mentions": true, // If turned on, filter out <USER_MENTION> tokens
        "filter_url": true, // If turned on, filter out any <URL> tokens
        "filter_retweet": true, // If turned on, filter out any <RETWEET> tokens
        "filter_stopwords": false, // If turned on, filter out stopwords before modeling.
        "keep_pronouns": true, // If turned on with filter_stopwords also on, preserve pronouns within the data set
        "emoji_handling": null, // Either null, "strip", or "replace". Strip removes, while replace makes an emoji <EMOJI> token
        "filter_mh_subreddits":"all", // Either null, "rsdd", "smhd", or "all". If subreddit metadata available, ignore subreddits in these lists
        "filter_mh_terms":null, // Either null, "rsdd", or "smhd". If specified, ignore posts that have terms from these lists in them.
        "max_tokens_per_document": null, // Select only the first n tokens from each document for modeling
        "max_documents_per_user": null, // Select only the n most recent documents from each user
        "preserve_case": false, // If turned on and preprocessed data not uncased, this preseves that capitatlization schema
        "max_vocab_size": null, // Select the k most common terms in the vocabulary
        "min_token_freq": 10, // Minimum occurence across data set to keep term in vocabulary
        "max_token_freq": null, // Maximum occurence across data set to keep term in vocabulary
        "ngrams": [1,1], // Min and max n-gram range
        "binarize_counter": true // If turned on, token frequencies are counted on a user-level. E.g. 10 mentions of a term from a single user only counts once toward its frequency.
    },
    "grid_search": true, // If turned on, train the models using a hyperparameter grid search
    "grid_search_kwargs":{ // High-level parameters for the grid search
        "config":"./configurations/hyperparameter_search/basic.json", // Path to the grid search hyperparameter configuration
        "test_size":0.2, // How much of the training data should be set aside as a development set within the grid search
        "dev_frac":0, // How much of the development data should be included in the grid search process
        "dev_in_vocab":false, // If turned on, let the development data be used to initialize the vocabulary
        "score_func":"f1", // Score function to maximimize within grid search
        "use_multiprocessing":false // For small data sets, turn on multiprocessing to get a speed up
    },
    "model": "logistic_regression", // If not using a grid search, choose the type of classifier to use
    "model_kwargs": { // Parameters passed to the sklearn classifier initialization
        "C": 1e3,
        "solver": "lbfgs",
        "random_state":42,
        "max_iter":1000,
        "class_weight":"balanced"
    },
    "preprocessing_kwargs": { // Choose which features will be used in the model, if not using a grid search
        "feature_flags":{
                        "bag_of_words":false, // Standard bag-of-words representation of term counts
                        "tfidf":true, // Apply TF-IDF to the bag-of-words representation
                        "glove":false, // Use pooled embeddings applied to the bag-of-words representation
                        "liwc":false, // Use LIWC distributions to represent the bag-of-words
                        "lda":false // Fit an LDA model to get a distribution over topics
                        },
        "feature_kwargs":{ // Hyperparameters passed to each of the feature transformers
                        "tfidf":{},
                        "glove":{
                                "dim":200,
                                "pooling":"mean"
                        },
                        "liwc":{
                                "norm":"max"
                               },
                        "lda":{
                                "n_components":50,
                                "doc_topic_prior":null,
                                "topic_word_prior":null,
                                "random_state":42
                        }
                    },
        "standardize":true // If turned on, center and scale the data set
    },
    "feature_selector": null, // Able to specifiy kldivergence, meaning to choose features with high information for each class
    "feature_selection_kwargs": { // Hyperparameters passed to mhlib.model.feature_selectors.KLDivergenceSelector
        "top_k": 20000,
        "min_support": 10,
        "stopwords": null,
        "add_lambda": 0.01,
        "beta": 0.1
    }
}
```

### Models

Parameters in `train_template.json` are largely the same as the domain transfer configuration, albeit only one domain is currently allowed to be specified.

### Hyperparameter Search

Files here can be used to specifiy parameters used for doing a grid search of model hyperparameters. The `_default.json` template is listed and explained below. The path to these files would be specified in the configurations from the aforementioned sections.

```
[ \\ List of hyperparameter sets in case you want to run multiple combinations
    {
        "name":"_default", // Arbitrary name for the set of hyperparameters searched over in this dictionary
        "feature_selector_ablation":false, // If turned on, consider models that both have and don't have a feature selector 
        "feature_set_ablation":true, // If turned on, incrementally add features to the model, choosing the best combination at each point
        "model":[
                { // Clasifier and parameters used as part of the grid search
                 "name":"logistic_regression",
                 "solver":["lbfgs"],
                 "max_iter":[1000],
                 "C":[1e-3,1e-2,0.01,1,10,100,1e3],
                 "class_weight":["balanced",null],
                 "random_state":[42]
                },
                {
                 "name":"svm",
                 "kernel":["linear","rbf"],
                 "C":[1e-3,1e-2,0.01,1,10,100,1e3],
                 "class_weight":["balanced",null],
                 "random_state":[42]
                 },
                 {
                    "name":"mlp",
                    "hidden_layer_sizes":[[100]],
                    "activation":["tanh","relu","logistic"],
                    "alpha":[1e-5,1e-4,1e-3,1e-2,0.01,1,10,100,1e3],
                    "batch_size":[40],
                    "learning_rate":["adaptive"],
                    "random_state":[42]
                    }                     
               ],
        "features":{ // Which features should be considered as part of the grid search, and what hyperparameters for those features should be searched over
            "bag_of_words":{},
            "tfidf":{},
            "glove":{
                "dim":[25, 50, 100, 200],
                "pooling":["mean"]
            },
            "liwc":{
                "norm":["max"]
            },
            "lda":{
                "n_components":[25, 50, 100, 200],
                "doc_topic_prior":[null],
                "topic_word_prior":[null]
            }
        },
        "standardize":[true], // Which standardization options should be considered as part of the grid search
        "feature_selector":{ // Which feature selector hyperparameters should be explored during the grid search
            "kldivergence":{
                "top_k":[10000,25000,50000,100000],
                "min_support":[10],
                "add_lambda":[0.01],
                "beta":[0.1]
            }
        }
    }
]
```