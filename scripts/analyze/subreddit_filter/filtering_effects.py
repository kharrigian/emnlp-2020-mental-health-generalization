
## Date Boundaries
MIN_DATE="2014-01-01"
MAX_DATE="2020-01-01"

###################
### Imports
###################

## Standard Library
import os
from functools import partial
from datetime import datetime

## Analysis Helpers
from scripts.analyze.subreddit_filter.filtering_effects_helpers import *

## Local Modules
from mhlib.util.logging import initialize_logger

###################
### Globals
###################

## Plotting Directory
PLOT_DIR = "./plots/subreddit_filter/"

## Initialize Logger
LOGGER = initialize_logger()

###################
### Plotting
###################

def plot_classification_results(results_dict):
    """

    """
    ## Plot Results
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bar_width = .95 / len(results_dict)
    for f, (fs, fs_res) in enumerate(results_dict.items()):
        for m, met in enumerate(["precision","recall","fscore"]):
            ax.bar(0.025 + m + bar_width * f,
                fs_res["mean"].loc["test"][met],
                yerr = fs_res["std"].loc["test"][met],
                width = bar_width,
                color = f"C{f}",
                alpha = 0.5,
                align = "edge",
                label = fs if m == 0 else "")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xticks(np.arange(3)+.5)
    ax.set_xlim(0,3)
    ax.set_xticklabels(["Precision","Recall","F-score"])
    fig.tight_layout()
    return fig, ax

def plot_coefficients(results_dict):
    """

    """
    ## Plot Coefficients
    fig, ax = plt.subplots(1, len(results_dict), figsize=(10,5.8))
    for f, (fs, fs_res) in enumerate(results_dict.items()):
        plot_data = fs_res["coefficients"].head(30).iloc[::-1]
        ax[f].barh(np.arange(len(plot_data)),
                plot_data.values)
        ax[f].set_yticks(np.arange(len(plot_data)))
        ax[f].set_yticklabels(plot_data.index.tolist(), fontsize=6)
        ax[f].set_title(fs, fontsize=6, loc="left")
        ax[f].set_ylim(-.5,29.5)
        ax[f].set_xlabel("Coefficient", fontsize=8)
    fig.tight_layout()
    return fig, ax

def show_null_users(cv_results):
    """

    """
    null_pivot = pd.pivot_table(cv_results,
                               index=["group"],
                               columns=["fold"],
                               values=["null_users"],
                               aggfunc=max)["null_users"]
    LOGGER.info(null_pivot)
    
###################
### Load Metadata
###################

## Load Wolohan Dataset
metadata_file = "./data/processed/reddit/wolohan/wolohan_metadata.csv"
metadata = pd.read_csv(metadata_file)

###################
### Subreddit Activity
###################

## Load Subreddits For Users
mp = Pool(8)
subreddit_loader = partial(get_subreddit_counts, min_date=MIN_DATE, max_date=MAX_DATE)
res = dict(tqdm(mp.imap_unordered(subreddit_loader, metadata["source"].tolist()), total=len(metadata), file=sys.stdout))
mp.close()

## Vectorize
filenames = sorted(res)
subreddits = sorted(set(flatten(res.values())))
sub2vec = create_dict_vectorizer(subreddits)
X = vstack(list(map(lambda f: sub2vec.transform(res[f]), filenames)))
y = metadata.set_index("source")["depression"].loc[filenames].values

## Class Distribution
X_binary = (X>0).astype(int)
dep_mask = np.where(y=="depression")[0]
con_mask = np.where(y=="control")[0]
counts = np.vstack([X_binary[dep_mask].sum(axis=0),
                    X_binary[con_mask].sum(axis=0)])
counts_df = pd.DataFrame(counts, index=["depression","control"], columns=subreddits).T

## Ratios
alpha = 1e-10
counts_df["pct_depression"] = (counts_df["depression"] + alpha) / ((y == "depression").sum() * (1 + alpha))
counts_df["pct_control"] = (counts_df["control"] + alpha) / ((y == "control").sum() * (1 + alpha))
counts_df["pct_total"] = (counts_df["control"] + counts_df["depression"] + alpha) / (len(y) * (1 + alpha))
counts_df["weighted_log_ratio"] = counts_df["pct_depression"] * np.log(counts_df["pct_depression"] / counts_df["pct_control"])
counts_df["pmi"] = np.log(counts_df["pct_depression"] / counts_df["pct_total"])
counts_df.sort_values("weighted_log_ratio",ascending=False,inplace=True)

###################
### Subredit Modeling (Subreddit Filters)
###################

LOGGER.info("~"*50 + "\nStarting Subreddit Modeling\n" + "~"*50)

## Filter Out Users That will Lose All Features
large_subreddit_mask = [i for i, s in enumerate(subreddits) if s.lower() not in filter_subs["All Mental Health"]]
large_subreddit_user_mask = np.nonzero(X[:, large_subreddit_mask].sum(axis=1)>0)[0]
X = X[large_subreddit_user_mask]
y = y[large_subreddit_user_mask]
filenames = [filenames[i] for i in large_subreddit_user_mask]

## Run Tests
subreddit_results = {}
for fs, fs_list in filter_subs.items():
    cv_results, avg_cv_results, std_cv_results, filtered_users, coefs = model_subreddits(X, y, subreddits, fs_list)
    LOGGER.info(f"\nUsers With Null Feature Sets: {fs}")
    show_null_users(cv_results)
    LOGGER.info("\n")
    subreddit_results[fs] = {"cv":cv_results,
                   "mean":avg_cv_results,
                   "std":std_cv_results,
                   "n_filtered":filtered_users,
                   "coefficients":coefs
                   }

## Number of Users With Everything Filtered
n_filtered = dict((x, y["n_filtered"]) for x, y in subreddit_results.items())
print(n_filtered)

## Create Plot Output Directory
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

## Plot Results
fig, ax = plot_classification_results(subreddit_results)
plt.savefig(f"{PLOT_DIR}wolohan_subreddit_filtering_subreddit_model_results.png", dpi=300)
plt.close()

## Plot Coefficients
fig, ax = plot_coefficients(subreddit_results)
plt.savefig(f"{PLOT_DIR}wolohan_subreddit_filtering_subreddit_model_coefs.png", dpi=300)
plt.close()

###################
### Language Modeling (Subreddit Filtering)
###################

LOGGER.info("~"*50 + "\nStarting Language Modeling (Subreddit Filter)\n" + "~"*50)

## Run Tests
subreddit_filter_language_results = {}
for fs, fs_list in filter_subs.items():
    cv_results, avg_cv_results, std_cv_results, filtered_users, coefs = model_language(filenames,
                                                                           metadata,
                                                                           filtered_subs=fs_list,
                                                                           filtered_language=set(),
                                                                           min_usage=10,
                                                                           min_date=MIN_DATE,
                                                                           max_date=MAX_DATE)
    LOGGER.info(f"\nUsers With Null Feature Sets: {fs}")
    show_null_users(cv_results)
    LOGGER.info("\n")
    subreddit_filter_language_results[fs] = {"cv":cv_results,
                            "mean":avg_cv_results,
                            "std":std_cv_results,
                            "n_filtered":filtered_users,
                            "coefficients":coefs
                            }

## Plot Results
fig, ax = plot_classification_results(subreddit_filter_language_results)
plt.savefig(f"{PLOT_DIR}wolohan_subreddit_filtering_language_model_results.png", dpi=300)
plt.close()

## Plot Coefficients
fig, ax = plot_coefficients(subreddit_filter_language_results)
plt.savefig(f"{PLOT_DIR}wolohan_subreddit_filtering_language_model_coefs.png", dpi=300)
plt.close()

###################
### Language Modeling (Language Filtering)
###################

LOGGER.info("~"*50 + "\nStarting Language Modeling (Language Filter)\n" + "~"*50)

## Run Tests
language_filter_language_results = {}
for fs, fs_list in filter_language.items():
    cv_results, avg_cv_results, std_cv_results, filtered_users, coefs = model_language(filenames,
                                                                           metadata,
                                                                           filtered_subs=set(),
                                                                           filtered_language=fs_list,
                                                                           min_usage=10,
                                                                           min_date=MIN_DATE,
                                                                           max_date=MAX_DATE)
    LOGGER.info(f"\nUsers With Null Feature Sets: {fs}")
    show_null_users(cv_results)
    LOGGER.info("\n")
    language_filter_language_results[fs] = {"cv":cv_results,
                            "mean":avg_cv_results,
                            "std":std_cv_results,
                            "n_filtered":filtered_users,
                            "coefficients":coefs
                            }

## Plot Results
fig, ax = plot_classification_results(language_filter_language_results)
plt.savefig(f"{PLOT_DIR}wolohan_language_filtering_language_model_results.png", dpi=300)
plt.close()

## Plot Coefficients
fig, ax = plot_coefficients(language_filter_language_results)
plt.savefig(f"{PLOT_DIR}wolohan_language_filtering_language_model_coefs.png", dpi=300)
plt.close()

###################
### Language Modeling (Language + Subreddit Filtering)
###################

LOGGER.info("~"*50 + "\nStarting Language Modeling (Both Filters)\n" + "~"*50)

## Run Tests
language_and_subreddit_filter_language_results = {}
for fs in ["None","RSDD","SMHD"]:
    cv_results, avg_cv_results, std_cv_results, filtered_users, coefs = model_language(filenames,
                                                                           metadata,
                                                                           filtered_subs=filter_subs[fs],
                                                                           filtered_language=filter_language[fs],
                                                                           min_usage=10,
                                                                           min_date=MIN_DATE,
                                                                           max_date=MAX_DATE)
    LOGGER.info(f"\nUsers With Null Feature Sets: {fs}")
    show_null_users(cv_results)
    LOGGER.info("\n")
    language_and_subreddit_filter_language_results[fs] = {"cv":cv_results,
                            "mean":avg_cv_results,
                            "std":std_cv_results,
                            "n_filtered":filtered_users,
                            "coefficients":coefs
                            }

## Plot Results
fig, ax = plot_classification_results(language_and_subreddit_filter_language_results)
plt.savefig(f"{PLOT_DIR}wolohan_language_and_subreddit_filtering_language_model_results.png", dpi=300)
plt.close()

## Plot Coefficients
fig, ax = plot_coefficients(language_and_subreddit_filter_language_results)
plt.savefig(f"{PLOT_DIR}wolohan_language_and_subreddit_filtering_language_model_coefs.png", dpi=300)
plt.close()

LOGGER.info("Script Complete.")