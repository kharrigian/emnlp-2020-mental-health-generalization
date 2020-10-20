# Data

- `raw`: Raw datasets (Twitter and Reddit) in their distributed format.
- `processed`: Standardized formats of data from each platform (e.g. tokenized, metadata)
- `resources`: External data resources (e.g. embeddings, dictionaries)
- `results`: Experimental results and cached data objects

## Raw Data Sources

We currently have data from Reddit and Twitter.

### Twitter

- `CLPsych 2015 Shared Task: Depression and PTSD on Twitter` ("clpsych" & "clpsych_deduped")
-- Data from the CLPsych 2015 shared task. All users are anonymized.
-- We no longer have perfect matches between the control users and the mental-health-status users. However, we are able to sample in a stratified manner based on age, gender, and test set distiction to come close to the original splits.

- `Multi-Task Learning for Mental Health using Social Media Text` ("multitask")
-- Data sourced in similar manner to CLPsych 2015 shared task (e.g. regular expressions).
-- Data set is expanded in size and contains more mental health disorders (e.g. anxiety, sucidial ideation, PTSD, Bipolar disorder)

### Reddit

- `Detecting Linguistic Traces of Depression in Topic-Restricted Text: Attenting to Self-Stigmatized Depression in NLP` ("wolohan")
-- Data collected on 2019-10-25 on Keith's local machine using RedditData API wrapper. 
-- History collection limited to 10k most recent unique authors who started a submission in AskReddit and Depression.
-- Initial filtering required at least 1,000 words (identified using simple `split` method) across comment body, submission title, and submission self-text.
-- 7k control, 6.7k depressed individuals

- `Depression and Self-Harm Risk Assessment in Online Forums` ("rsdd")
-- Data downloaded from Georgetown dump after signing a usage agreement.
-- Data contains raw text posts, user labels, and the date the text was posted. No subreddit information is contained within and all users are anonymized.

- `SMHD: A Large-Scale Resource for Exploring Online Language Usage for Multiple Mental Health Conditions` ("smhd")
-- Data downloaded from Georgetown dump after signing a usage agreement.
-- Like RSDD, data contains raw text posts, user labels, and the date the text was posted. No subreddit information is contained within and all users are anonymized.
-- This data set shares overlap with RSDD, but is designed to include more mental health statuses (e.g. PTSD, Bipolar Disorder) and also to have higher precision labels.

## Processed

Processed data is broken down first by domain (e.g. Reddit vs. Twitter) and then by data set source (e.g. RSDD, SMHD, Topic-Restricuted and QNTFY). Note that QNTFY generally refers to data from Glen Coppersmith's research (e.g. CLPsych Shared Task and Multi-Task Learning). Consult code in `scripts/preprocess` to transform the raw, distributed formats of each dataset into these standardized formats for model training and evaluation.

## Resources

We use various resources as part of the data processing pipeline.

* `glove.twitter*` - Pretrained Twitter embeddings downloaded from Stanford website. Used in our modeling process to do mean-pooled representations of user-histories.
* `liwc2007.dic` - Mapping between regular-expression prefixes and LIWC dimensions (the 2007 version). Used to transform bag-of-words feature representations into a much lower dimension (64) using `mhlib.model.feature_extractors.LIWCTransformer`
* `mh_terms.json` - Keywords and regular expressions used in the RSDD and SMHD data sets to filter out explicit mental health content from analysis.
* `mh_subreddits.json` - Subreddits identified by RSDD and SMHD datasets, along with an indepenent analysis, that are strongly associated with mental health (e.g. depression-support).

## Results

Results (data, model, some visualizations) are stored in this directory.

* `domain_transfer/` - Results from domain-transfer experiments.
* `temporal_effects/` - Results from temporal effects experiments.
* `cache/` - Intermediary files and results from one-off-scripts found in `scripts/analyze/`
