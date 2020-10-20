# Do Models of Mental Health Based on Social Media Data Generalize?

This repository contains code used to explore the reliability and robustness of mental-health status predictive models for social media. Much of the data used as part of this project requires IRB approval and signed data usage agreements. The associated code library (`mhlib`) works as a standalone offering in many cases. However, much of the code in `scripts/` is currently set up under the assumption of working on the JHU CLSP grid. At the minimum, you would have to update data set paths to run the code. At the maximum, you have to replace scheduler calls with subprocesses et al. Please note that commerical use of any code or data in this repository is strictly prohibited. If you use any code or data, please cite our paper using the bibliographic information below.

```
@inproceedings{harrigian2020mentalhealth,
  title={Do Models of Mental Health Based on Social Media Data Generalize?},
  author={Harrigian, Keith; Aguirre, Carlos; Dredze, Mark},
  year={2020}
}
```

## Contact

If you have questions about code or data, please contact Keith Harrigian (kharrigian@jhu.edu).

## Setup and Installation

To run most of the code, you will need to begin by installing the repositories library of functions. We recommend using the following command to install the development version of the library. Note that the code was developed using Python 3.7+. To ensure reproducibility, the package will not install if any other version of Python is listed as the environment's primary version.

```
pip install -e .
```

If you wish to collect Reddit data using the Python Reddit API Wrapper (PRAW), you will need to provide API credentials for the `mhlib` package to reference. To do so, rename `config_template.json` as `config.json` and fill in all respective fields of the JSON file. Note, you can still acess Reddit data without official API access, instead via PSAW--a wrapper around the Pushshift.io database. At the current moment, tests associated with the Reddit API will fail without valid credentials.

### Additional Downloads

**GloVe**

Download the Twitter-based word embeddings here: https://nlp.stanford.edu/projects/glove/. Save them to the "data/resources/" folder.

**Linguistic Inquiry and Word Count (LIWC)**

You will need the 2007 version of the LIWC dictionary to use LIWC-based features. Save them as "data/resources/liwc2007.dic".

**NLTK Corpora**

After installing `nltk`, you should open a shell and call `import nltk; nltk.download()`. This command will open a GUI which should be used to download "all corpora".

**Demoji Code Dictionary**

Some of our analysis converts unicode emojis into interpretable ASCII strings via the `demoji` library. This library relies on an internal dictionary that must be downloaded as follows: `import demoji; demoji.download_codes()`.

## Testing

To ensure the `mhlib` library has been properly installed, you may run the test suite using the following code. Note that tests for the Reddit API will fail intermittently if you have not included valid crentials in a `config.json` file in the root of this repository.

```
pytest tests/ -Wignore -v
```

Alternatively, to run tests and check package coverage, you can do so using

```
pytest tests/ --cov=mhlib/ --cov-report=html -v -Wignore
```

## Data Access

This project evaluates (or plans to evaluate) the list of datasets below. Other than the "Topic-Restricted Text" dataset, all datasets require data access agreements from their respective providers.

#### Twitter

1. `CLPsych 2015` - "CLPsych 2015 Shared Task: Depression and PTSD on Twitter". For more information, contact Mark Dredze (mdredze@cs.jhu.edu).
2. `Multi-Task Learning` - "Multi-Task Learning for Mental Health using Social Media Text". For more information, contact Glen Coppersmith (glen@qntfy.com).

#### Reddit

1. `Topic-Restricted Text (Wolohan)` - "Detecting Linguistic Traces of Depression in Topic-Restricted Text: Attenting to Self-Stigmatized Depression in NLP". For more information, see `scripts/acquire/get_wolohan.py`.
2. `RSDD` - "Depression and Self-Harm Risk Assessment in Online Forums". For more information, see http://ir.cs.georgetown.edu/resources/rsdd.html.
3. `SMHD` - "SMHD: A Large-Scale Resource for Exploring Online Language Usage for Multiple Mental Health Conditions". For more infroatmion, see http://ir.cs.georgetown.edu/resources/smhd.html.

## Pipeline

There are 4 primary steps in the modeling/experimention pipeline. To reproduce results, one can follow this order of operations. All code is expected to be run from the root directory of the repository. Note that some paths are set-up assuming access to the CLSP grid. If you need help adapting code to your local computing environment, please reach out and we'll try to help!

1. `scripts/acquire` - Scripts in the directory retrieve the `Topic-Restricted Text` and properly concatenate the first two Twitter datasets (which are sourced from QNTFY and Johns Hopkins University).
2. `scripts/preprocess` - Scripts in this directory create tokenized and user-separated data files that can be used for modeling. The `prepare_*.py` scripts should be run first. `scripts/model/compile_metadata_lookup.py` can be executed after all preparation scripts are complete to generate summarized metadata CSV files.
3. `scripts/experiment` - Train ML classifiers for various experiments (e.g. domain transfer, etc.). The primary source of code used in Section 5 of our paper.
4. `scripts/analyze` - Run additional analysis of results from the prior modeling phase. Mostly one-off explorations of each dataset.
5. `scripts/model` - Train a new classification model, optionally running cross-validation. Alternatively, apply a trained classification model to new data.