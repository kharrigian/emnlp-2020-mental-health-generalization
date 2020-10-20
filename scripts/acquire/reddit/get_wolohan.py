
## Standard Data Directory
DATA_DIR = "./data/raw/reddit/wolohan/"

######################
### Imports
######################

## Standard Libary
import os
import datetime
from glob import glob

## External Libaries
import pandas as pd
import joblib

## Local
from mhlib.acquire.reddit import RedditData

######################
### Output Setup
######################

## Create Data Directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

######################
### Helpers
######################

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.

    Args:
        l (list): List of objects
        n (int): Chunksize
    
    Yields:
        Chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

######################
### Collection
######################

"""
Method:
    - Get 10,000 most recent post authors from depression and AskReddit
      subreddit
    - Scrape entire post history of users
    - Filter out users with less than 1000 words used across 
      comments and submissions
"""

## Parameters
subreddit_queries = ["depression", "AskReddit"]
min_words = 1000
n_authors = 10000
prefix = "wolohan"
author_chunksize = 20

## Initial Date (Working Backwards)
current_date = datetime.datetime.now()
end_date = current_date
start_date = end_date - datetime.timedelta(weeks=1)

## Initialize Reddit Data Class
reddit = RedditData()

## Function To Meet Word Minimums
def filter_by_word_count(author_comments,
                         author_submissions,
                         min_words):
    """
    Identify set of authors meeting minimum word count 
    requirement

    Args:
        author_comments (dataframe): Author comment history
        author_submissions (dataframe): Author submission history
        min_words (int): Minimum number of words across history
    
    Returns:
        filtered_authors (set): Set of authors meeting criteria
    """
    ## Helper Function
    wcount = lambda i: len(str(i).split()) if i is not None else 0
    get_count_series = lambda df, col, authors: df.groupby(["author"])[col].sum().reindex(authors).fillna(0).rename("count")
    ## Count Words (Naive)
    author_comments["word_count"] = author_comments.body.map(wcount)
    author_submissions["title_word_count"] = author_submissions.title.map(wcount)
    author_submissions["selftext_word_count"] = author_submissions.selftext.map(wcount)
    ## Aggregate Across Authors
    authors = list(set(author_comments.author) | set(author_submissions.author))
    author_wc = get_count_series(author_comments,
                                 "word_count",
                                 authors) + \
                get_count_series(author_submissions,
                                 "title_word_count",
                                 authors) + \
                get_count_series(author_submissions,
                                 "selftext_word_count",
                                 authors)
    ## Drop Added Columns
    author_comments.drop(["word_count"],axis=1,inplace=True)
    author_submissions.drop(["title_word_count"],axis=1,inplace=True)
    author_submissions.drop(["selftext_word_count"],axis=1,inplace=True)
    ## Filter
    filtered_authors = author_wc.loc[author_wc>=min_words].index.values
    return set(filtered_authors)

## Construct Queries for Each Subreddit
for s, subreddit in enumerate(subreddit_queries):
    ## Identify Submission Cache File
    sub_outfile = f"{DATA_DIR}{prefix}_{subreddit}_submissions.joblib"
    if os.path.exists(sub_outfile):
        subreddit_posts = joblib.load(sub_outfile)
    else:
        ## Collect Submissions
        subreddit_posts = pd.DataFrame()
        author_count = 0
        while author_count < n_authors:
            submissions = reddit.retrieve_subreddit_submissions(
                                subreddit=subreddit,
                                start_date=start_date.date().isoformat(),
                                end_date=end_date.date().isoformat(),
                                limit=100000
            )
            end_date = start_date
            start_date = end_date - datetime.timedelta(weeks=1)
            subreddit_posts = subreddit_posts.append(submissions)
            print("Current Author Count for r/{}: {}/{}".format(
                subreddit,
                len(subreddit_posts.author.unique()),
                n_authors
            ))
            author_count = len(subreddit_posts.author.unique())
        subreddit_posts = subreddit_posts.reset_index(drop=True)
        ## Save Submissions
        print(f"Saving Submissions for {subreddit}")
        _ = joblib.dump(subreddit_posts,
                        sub_outfile,
                        compress=("gzip", 5))
    ## Drop Deleted Authors
    subreddit_posts = subreddit_posts.loc[(subreddit_posts.author!="[deleted]")&
                                          (subreddit_posts.author!="[removed]")]
    ## Identify Most Recent Authors
    submission_authors = subreddit_posts.sort_values("created_utc",
                                                     ascending=False).\
                                         drop_duplicates("author",
                                                         keep="first").\
                                        author.values[:n_authors]
    ## Group Authors and collect histories
    author_groups = list(chunks(submission_authors, author_chunksize))
    hist_outdir = f"{DATA_DIR}{prefix}_{subreddit}_histories/"
    if not os.path.exists(hist_outdir):
        os.makedirs(hist_outdir)
    for a, author_group in enumerate(author_groups):
        print("Collecting Histories From Group {}/{}".format(a+1, len(author_groups)))
        ## Filter Out Bat Authors
        author_group = list(filter(lambda x: x != "[deleted]" and x != "[removed]", author_group))
        ## Identify Filenames And Check For Existence
        hist_com_outfile = f"{hist_outdir}comment_chunk_{a+1}.joblib"
        hist_sub_outfile = f"{hist_outdir}submission_chunk_{a+1}.joblib"
        if os.path.exists(hist_com_outfile) and os.path.exists(hist_sub_outfile):
            continue
        ## Retrieve Data
        author_comments = reddit.retrieve_author_comments(
                    author=author_group,
                    end_date=current_date.date().isoformat(),
                    limit=None
        )
        author_submissions = reddit.retrieve_author_submissions(
                    author=author_group,
                    end_date=current_date.date().isoformat(),
                    limit=None
        )
        ## Filter By Word Count
        authors_to_keep = filter_by_word_count(author_comments,
                                               author_submissions,
                                               min_words)
        author_comments = author_comments.loc[author_comments.author.isin(authors_to_keep)]
        author_submissions = author_submissions.loc[author_submissions.author.isin(authors_to_keep)]
        author_comments = author_comments.reset_index(drop=True)
        author_submissions = author_submissions.reset_index(drop=True)
        ## Save Histories
        _ = joblib.dump(author_comments,
                        hist_com_outfile,
                        compress=("gzip", 5))
        _ = joblib.dump(author_submissions,
                        hist_sub_outfile,
                        compress=("gzip", 5))

## Create Label Mapping (Author:Label, depression takes precedence)
all_submissions = []
for sub in subreddit_queries:
    all_submissions.append(joblib.load(f"{DATA_DIR}{prefix}_{sub}_submissions.joblib"))
all_submissions = pd.concat(all_submissions).reset_index(drop=True)
label_mapper = lambda x: "depression" if "depression" in set(x) else "control"
label_map = all_submissions.groupby(["author"]).agg({"subreddit":label_mapper})
_ = joblib.dump(label_map.subreddit.to_dict(),
                f"{DATA_DIR}{prefix}_label_map.joblib",
                compress=("gzip",5))

## View Amount of Data in Histories
dataset_size = {"comments":{},"submissions":{}}
for sub in subreddit_queries:
    for sub_f in glob(f"{DATA_DIR}{prefix}_{sub}_histories/*"):
        is_submission = os.path.basename(sub_f).startswith("submission")
        df = joblib.load(sub_f)
        if is_submission:
            dataset_size["submissions"].update(df.author.value_counts().to_dict())
        else:
            dataset_size["comments"].update(df.author.value_counts().to_dict())

## Look at Label Distribution
label_map_filtered = dict((author, label) for author, label in label_map.subreddit.items() \
                          if author in dataset_size["comments"] or \
                             author in dataset_size["submissions"])
print("Label Distribution:")
print(pd.Series(label_map_filtered).value_counts())