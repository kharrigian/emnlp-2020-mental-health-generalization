# Scripts

High-level code for training and evaluating mental-health classifiers.

### Acquire

This folder contains a couple of scripts for retrieving user-level mental health status labels.

* `twitter/get_qntfy.py` - Parses data from the CLPSych 2015 Shared Task and Multi-Task Learning for Mental Health. Merges the two data sets as well into a merged data set that deduplicates users between the two datasets.
* `reddit/get_wolohan.py` - Attempts to create a dataset similar to that of the Wolohan et al. paper. Essentially pulls users who posted in r/Depression and then gets a similarly-sized sample of users who have posted in r/AskReddit (but not in r/Depression).

### Preprocess

Code here is used to transform raw comment and tweet data into the format the model expects. Primarily, these scripts will tokenize all text and create a list of JSON dictionaries for each comment associated with a user. Once all preprocessing is complete, we also have provided a script to create CSV files matching preprocessed files to mental-health status labels.

* `twitter/prepare_qntfy.py` - Format the CLPSych 2015 Shared Task and Multi-Task Learning for Mental Health data. Outputs formatted files to `data/processed/twitter/qntfy/`
* `reddit/prepare_wolohan.py` - Format the Topic-Restricted Tet data. Outputs formatted files to `data/processed/reddit/wolohan/`
* `reddit/prepare_rsdd.py` - Format the RSDD data. Outputs formatted files to `data/processed/reddit/rsdd/`
* `reddit/prepare_smhd.py` - Format the SMHD data. Outputs formatted files to `data/processed/reddit/smhd/`

Once all preprocessing scripts have been run, users can run `scripts/model/compile_metadata_lookup.py` to generate CSV files that map mental health condition labels to processed data files (and include some additional metadata).

### Analyze

One-off scripts used to examine each of the data sets and run minor experiments.

**Feature Analysis**
* `features/vocabulary_similarity.py` - Explores overlap in vocabulary and similarity in post distributions.
* `features/representative_unigrams.py` - Uses KL-divergence to get most representative unigrams from each class.
* `features/external_resource_distributions.py` - Examines semantic differences in data sets based on LIWC and GloVe
* `features/liwc.py` - Examines stability of LIWC across datasets and as a facet of constraining vocabularies

**Subreddit Filtering Effects**
* `subreddit_filter/filtering_effects.py` - Explores the effect of filtering out data from certain subreddits and data that includes certain keywords (within the Wolohan data set). This script loads in code from `subreddit_filter/filtering_effects_helpers.py`

**Temporal Analysis**
* `temporal/post_distribution_comparison.py` - Identifies the distribution of posts over time for each of the data sets to help show-case there may be temporal confounds in our results.
* `temporal/post_distribution.py` - Like the comparison script, but operates on a single dataset/condition.
* `temporal/sample_size_by_threshold.py` - Identifies the number of individuals available per time period under a variety of thresholds.

### Experiment

Large-scale experiments primarily for domain-transfer evaluation.

**Domain Transfer**

*Summary*: Train a model on one dataset and apply it to another (with options to run cross-validation).

* `domain_transfer/domain_transfer.py` - Cross-validation procedure with the option to fit a model using one data set and apply it to another.
* `domain_transfer/domain_transfer_wrapper.py` - Run `domain_transfer.py` for several combinations of data sets, setting up unique jobs on the CLSP grid to parallelize the training process.
* `domain_transfer/process_domain_transfer_results.py` - Score model performance across data sets output from `domain_transfer_wrapper.py`. Currently a minimal offering that prints classification metrics in an easy-to-digest table.

**Temporal Effects**

*Summary*: Train a model on one time period and apply it to another (within- and across-domains).

* `temporal_effects/temporal_effects_wrapper.py` - Set up multiple experiments of temporal-transfer, preserving models to reduce runtime.
* `temporal_effects/temporal_effects.py` - Run a domain-transfer experiment with added effect of time dynamics.
* `temporal_effects/temporal_effects_helper.py` - Run as independent jobs scheduled by `temporal_effects.py`.
* `temporal_effects/temporal_effects_results.py` - Create plots summarizing temporal transfer performance for several datasets at once.

### Model

Standalone scripts for training/evaluating mental health classifiers.

**Prerequisites**
* `compile_metadata_lookup.py` - Expects that all of the `prepare_*` scripts above in the `preprocess` directory have been run. Creates a file (e.g. `data/processed/reddit/rsdd/rsdd_metadata.csv.`) that contains mappings between user processed files and mental health status labels for each data set.

**Training**
* `train.py` - Train a mental-health classifier and evaluate its test-set performance. Optionally, run cross-validation within the training data set to understand classification variance.

**Model Diagnostics**
* `diagnostics/examine_model.py` - Create visualizations to help interpret model coefficients.
* `diagnostics/precision_recall_curve.py` - Generate precision/recall curves for trained models.

**Inference**
* `preprocess.py` - Provides a framework for preprocessing raw data from Twitter/Reddit and transforming it into standardized format ingested by classifiers.
* `infer.py` - Ingests preprocessed data and makes mental health status inferences using a pretrained model.
* `evaluate.py` - Applies trained model to existing set of data set splits and evaluates predictive performance.