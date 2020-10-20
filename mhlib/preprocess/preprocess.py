
"""
Default preprocessing code for all model training.
"""

#############################
### Imports
#############################

## Local Modules
from .tokenizer import Tokenizer

#############################
### Globals
#############################

## Initialize Text Tokenizer
tokenizer = Tokenizer(stopwords=None,
                      keep_case=True,
                      negate_handling=True,
                      negate_token=True,
                      upper_flag=True,
                      keep_punctuation=True,
                      keep_numbers=True,
                      expand_contractions=True,
                      keep_user_mentions=True,
                      keep_pronouns=True,
                      keep_url=True,
                      keep_hashtags=True,
                      keep_retweets=True,
                      emoji_handling=None,
                      strip_hashtag=False)

#############################
### Field Schemas
#############################

## Fields Cached For Each Dataset
db_schema = {
            "qntfy":{
                            "tweet":{   ## Tweet Information
                                        "user_id_str":"user_id_str",
                                        "created_utc":"created_utc",
                                        "text":"text",
                                        "text_tokenized":"text_tokenized",
                                        "id_str":"tweet_id",
                                        ## User Metadata
                                        "age":"age",
                                        "gender":"gender",
                                        ## Labels (All Possible)
                                        "depression":"depression",
                                        "anxiety":"anxiety",
                                        "suicide_attempt":"suicide_attempt",
                                        "suicidal_ideation":"suicidal_ideation",
                                        "eating_disorder":"eating_disorder",
                                        "panic":"panic",
                                        "schizophrenia":"schizophrenia",
                                        "bipolar":"bipolar",
                                        "ptsd":"ptsd",
                                        ## Processing Metadata
                                        "entity_type":"entity_type",
                                        "date_processed_utc":"date_processed_utc",
                                        "source":"source",
                                        "dataset":"dataset",
                                    }
            },
            "rsdd":{
                    "comment":{ 
                                ## Post Information
                                "author":"user_id_str",
                                "created_utc":"created_utc",
                                "text":"text",
                                "text_tokenized":"text_tokenized",
                                ## Labels
                                "depression":"depression",
                                ## Processing Metadata
                                "entity_type":"entity_type",
                                "date_processed_utc":"date_processed_utc",
                                "source":"source",
                                "dataset":"dataset"
                                }
            },
            "smhd":{
                    "comment":{
                                ## Comment Information
                                "author":"user_id_str",
                                "created_utc":"created_utc",
                                "text":"text",
                                "text_tokenized":"text_tokenized",
                                ## Labels
                                "adhd":"adhd",
                                "anxiety":"anxiety",
                                "autism":"autism",
                                "bipolar":"bipolar",
                                "depression":"depression",
                                "eating":"eating_disorder",
                                "ocd":"ocd",
                                "ptsd":"ptsd",
                                "schizophrenia":"schizophrenia",
                                ## Processing Metadata
                                "entity_type":"entity_type",
                                "date_processed_utc":"date_processed_utc",
                                "source":"source",
                                "dataset":"dataset"
                                }
            },
            "wolohan": {
                        "comment":{
                                    ## Comment Information
                                    "author":"user_id_str",
                                    "subreddit":"subreddit",
                                    "created_utc":"created_utc",
                                    "body":"text",
                                    "id":"comment_id",
                                    "text_tokenized":"text_tokenized",
                                    "link_id":"submission_id",
                                    ## Labels
                                    "depression":"depression",
                                    ## Processing Metadata
                                    "entity_type":"entity_type",
                                    "date_processed_utc":"date_processed_utc",
                                    "source":"source",
                                    "dataset":"dataset"
                                    },
                        "submission":{
                                    ## Submission Information
                                    "author":"user_id_str",
                                    "subreddit":"subreddit",
                                    "created_utc":"created_utc",
                                    "selftext":"text",
                                    "title":"title",
                                    "id":"submission_id",
                                    "text_tokenized":"text_tokenized",
                                    "title_tokenized":"title_tokenized",
                                    ## Labels
                                    "depression":"depression",
                                    ## Processing Metadata
                                    "entity_type":"entity_type",
                                    "date_processed_utc":"date_processed_utc",
                                    "source":"source",
                                    "dataset":"dataset"
                                    }
                },      
}

#############################
### Functions
#############################

def _clean_surrogate_unicode(text):
    """
    Clean strings that have non-processable unicode (e.g.
    'RT @lav09rO5KgJS: Tell em J.T. ! 😂😍\ud83dhttp://t.co/Tc_qbFYmFYm')

    Args:
        text (str): Input text
    
    Returns:
        cleaned text if a unicode error otherwise arises
    """
    if text is None:
        return text
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError as e:
        if e.reason == 'surrogates not allowed':
            return text.encode("utf-8", "ignore").decode("utf-8")
        else:
            raise(e)

def format_tweet_data(data):
    """
    Extract tweet and user data from JSON dictionary
    
    Args:
        data (dict): Dictionary of raw tweet data
    
    Returns:
        formatted_data (dict): Data with user-level information extracted
    """
    ## Define Data To Extract (Tweet- and User-level)
    tweet_cols = ["truncated",
                  "text",
                  "in_reply_to_status_id",
                  "id",
                  "favorite_count",
                  "retweeted",
                  "in_reply_to_screen_name",
                  "in_reply_to_user_id",
                  "retweet_count",
                  "id_str",
                  "geo",
                  "in_reply_to_user_id_str",
                  "lang",
                  "created_at",
                  "in_reply_to_status_id_str",
                  "place"]
    user_cols = ["verified",
                 "followers_count",
                 "utc_offset",
                 "statuses_count",
                 "friends_count",
                 "geo_enabled",
                 "screen_name",
                 "lang",
                 "favourites_count",
                 "url",
                 "created_at",
                 "time_zone",
                 "listed_count",
                 "id_str",
                 "name",
                 "location",
                 "description",
                 "time_zone",
                 "utc_offset",
                 "protected",
                 "profile_image_url"]
    ## Extract Data
    formatted_data = {}
    for t in tweet_cols:
        if t in data:
            if t.startswith("in_reply"):
                formatted_data[t] = str(data[t])
            else:
                formatted_data[t] = data[t]
        else:
            formatted_data[t] = None
    if "full_text" in data:
        formatted_data["text"] = data["full_text"]
    if "extended_tweet" in data and "full_text" in data.get("extended_tweet"):
        formatted_data["text"] = data.get("extended_tweet").get("full_text")
    for u in user_cols:
        if "user" not in data:
            formatted_data[f"user_{u}"] = None
        else:
            if u in data["user"]:
                formatted_data[f"user_{u}"] = data["user"][u]
            else:
                formatted_data[f"user_{u}"] = None
    ## Clean Surrogate Unicode if Necessary
    formatted_data["text"] = _clean_surrogate_unicode(formatted_data["text"])
    return formatted_data
