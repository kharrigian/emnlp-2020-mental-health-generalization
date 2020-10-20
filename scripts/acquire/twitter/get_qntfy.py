
"""
Create a merged label file for all QNTFY data. Note that this
version of the script assumes access to the "reverse engineered"
version of raw data that was previously lost due to a RAID failure.
"""

## Helpful Paths
DATA_DIR = "./data/raw/twitter/"
QNTFY_PATH=f"{DATA_DIR}qntfy/"

###################
### Imports
###################

## Standard Library
import os
import json
from glob import glob

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

###################
### Load Deduped Labels
###################

## Load Multi-task Paper Labels (De-duplicated)
deduped_file = "neurotypical-anxiety-depression-suicide_attempt-eating-panic-schizophrenia-bipolar-ptsd-gender_human.DEDUPLCIATED.tsv"
deduped_labels = pd.read_csv(f"{QNTFY_PATH}{deduped_file}",
                             sep="\t",
                             header=None)
col_names = {0:"neurotypical",
             1:"anxiety",
             2:"depression",
             3:"suicide_attempt",
             4:"eating_disorder",
             5:"panic",
             6:"schizophrenia",
             7:"bipolar",
             8:"ptsd",
             9:"gender",
             10:"filepath"}
deduped_labels.rename(columns = col_names, inplace=True)

## Get User Identifier
deduped_labels["user_id_str"] = deduped_labels["filepath"].str.split("/").map(lambda i: i[-1])

## Check Filepath Existence
deduped_labels["source"] = deduped_labels["filepath"].map(lambda i:f"{i}.tweets.gz")
deduped_labels["source"] = deduped_labels["source"].map(os.path.abspath)

## Binarize Disorder Labels
disorders = []
for c in range(9):
    disorders.append(col_names[c])
    deduped_labels[col_names[c]] = deduped_labels[col_names[c]].map(lambda i: not i.startswith("NOT_")).astype(int)

## Drop Gender Column
deduped_labels.drop(["gender","filepath"], axis=1, inplace=True)

###################
### Multitask Data Set
###################

## Load Labels
multitask_nn_labels = []
for f in tqdm(glob(f"{QNTFY_PATH}tweets/*.json")):
    f_data = json.load(open(f,"r"))
    if "matched_control" in f_data:
        f_data["matched_control"] = str(f_data["matched_control"])
    if "matched_condition" in f_data:
        f_data["matched_condition"] = str(f_data["matched_condition"])
    f_data["user_id_str"] = str(f_data["user_id_str"])
    f_data["filepath"] = f.replace(".json",".tweets.gz")
    multitask_nn_labels.append(f_data)
multitask_nn_labels = pd.DataFrame(multitask_nn_labels)

## Mapping Between Conditions (At Least Ones Referenced in the Multitask Paper. Note that there Are Actually More)
condition_map = {
    "anxiety":"anxiety",
    "bipolar":"bipolar",
    "borderline":"borderline",
    "depression":"depression",
    "eating":"eating_disorder",
    "eating_disorder":"eating_disorder",
    "panic":"panic",
    "ptsd":"ptsd",
    "schizophrenia":"schizophrenia",
    "si":"suicidal_ideation",
    "suicidal":"suicidal_ideation",
    "suicidal_ideation":"suicidal_ideation",
    "suicide_attempt":"suicide_attempt",
}

## Replace Null Conditions With Empty Lists (e.g. Controls where appropriate)
multitask_nn_labels["conditions"] = multitask_nn_labels.apply(lambda row: [] if not pd.isnull(row["matched_condition"]) else row["conditions"],axis=1)

## Drop Null
multitask_nn_labels = multitask_nn_labels.dropna(subset=["conditions"]).copy()

## Remap Conditions
replace_conditions = lambda l: list(set([condition_map[i] for i in l if i in condition_map]))
multitask_nn_labels["conditions"] = multitask_nn_labels["conditions"].map(replace_conditions)

## Subset
multitask_nn_labels = multitask_nn_labels[["user_id_str","matched_condition","matched_control","conditions","filepath","wwbp_age","wwbp_gender","annotator"]]
multitask_nn_labels = multitask_nn_labels.reset_index(drop=True).copy().rename(columns={
                            "filepath":"source","wwbp_age":"age","wwbp_gender":"gender"
})

## Source
multitask_nn_labels["source"] = multitask_nn_labels["source"].map(os.path.abspath)

## Condition Expansion
all_conditions = sorted(set(condition_map.values()))
for condition in all_conditions:
    multitask_nn_labels[condition] = multitask_nn_labels["conditions"].map(lambda i: condition if condition in i else f"NOT_{condition}")
multitask_nn_labels["neurotypical"] = multitask_nn_labels["matched_condition"].map(lambda i: "neurotypical" if not pd.isnull(i) else "NOT_neurotypical")
all_conditions.append("neurotypical")

## Binarize Disorder Labels
for c in all_conditions:
    multitask_nn_labels[c] = multitask_nn_labels[c].map(lambda i: not i.startswith("NOT_")).astype(int)

## Add User Control, Drop Duplicates and Users Without Matches
multitask_nn_labels["match"] = multitask_nn_labels[["matched_condition","matched_control"]].apply(lambda row: row.dropna()[0],axis=1).astype(str)
anomalies = ['5940145065125009127', '4882443411083255938', '1964388080325920784'] # should be controls (duplicates)
multitask_nn_labels = multitask_nn_labels.loc[~multitask_nn_labels.matched_condition.isin(anomalies)]
multitask_nn_labels = multitask_nn_labels.loc[multitask_nn_labels.match.isin(multitask_nn_labels.user_id_str)]
multitask_nn_labels = multitask_nn_labels.loc[multitask_nn_labels[all_conditions].sum(axis=1) > 0]
multitask_nn_labels = multitask_nn_labels.loc[multitask_nn_labels.match.isin(multitask_nn_labels.user_id_str)]
multitask_nn_labels = multitask_nn_labels.reset_index(drop=True).copy()
multitask_nn_labels.drop(["matched_condition","matched_control"],axis=1,inplace=True)

## Comorbitity
multitask_comorbidity = pd.DataFrame(np.matmul(multitask_nn_labels[disorders].T.values, multitask_nn_labels[disorders].values),
                                     index=disorders,
                                     columns=disorders)

multitask_nn_labels.drop(["conditions","annotator"], axis=1, inplace=True)

###################
### CLPsych 2015 Data
###################

## Load CLPsych Label Files (Train/Test Splits + Full Label Set)
clp_train = pd.read_csv(f"{DATA_DIR}clpsych_2015_shared_task/anonymized_user_info_by_chunk_training.csv")
clp_test = pd.read_csv(f"{DATA_DIR}clpsych_2015_shared_task/anonymized_user_info_by_chunk_testing.csv")
clp_df = pd.read_csv(f"{DATA_DIR}clpsych_2015_shared_task/anonymized_user_info_by_chunk.csv")

## Rename User ID Column
for df in [clp_train, clp_test, clp_df]:
    df.rename(columns = {"anonymized_screen_name":"user_id_str"},inplace=True)

## Identify Train/Test/Unknown Split (in Shared Task)
clp_df.loc[clp_df.user_id_str.isin(clp_train.user_id_str), "split"] = "train"
clp_df.loc[clp_df.user_id_str.isin(clp_test.user_id_str), "split"] = "test"
clp_df.dropna(subset=["split"], inplace=True)
clp_df = clp_df.loc[clp_df["condition"].isin(set(["depression","ptsd","control"]))].copy()

## Append Source File
clp_df["source"] = clp_df.apply(lambda row: "{}tweets/{}.tweets.gz".format(
                                    QNTFY_PATH,
                                    row["user_id_str"]), axis=1)
clp_df["source"] = clp_df["source"].map(os.path.abspath)

## Expand Conditions
for cnow, clbl in zip(["control","depression","ptsd"],["neurotypical","depression","ptsd"]):
    clp_df[clbl] = (clp_df["condition"]==cnow).astype(int)

## Drop Unused Columns
clp_df.drop(["num_tweets","chunk_index","condition"],axis=1,inplace=True)

###################
### Merge and Reconcile
###################

"""
Notes:
- All users in multitask_nn_labels are in deduped_labels
- All but 80 users in clp_df are in deduped_labels
- No users in clp_df are in multitask_nn_labels
- 156 users from deduped_labels are not in clp_df or multitask_nn_labels. 64 of these do not 
  have any associated disorders. The remaining 92 were labeled neurotypical
-> Need 4 Data Splits CLPsych, Multitask (Full), Multitask (CLPsych De-duped), Merged
    CLPsych: All users in the CLPsych data
    CLPsych (Multitask De-duped): CLPsych Data Without the 80 users not in Multitask (Likely duplicates)
    Multitask (Full): All users in the Multitask Label Set
    Merged: De-duped Data Without Modification, Combines CLPsych with Multitask and Deduplicates Users
-> Users in CLPsych Test Set Don't Have Labels in Deduped, We Should Use CLPsych Labels. All labels in 
   multitask_disorders match thse in deduped_disorders
"""

## Drop Users in De-duped that Do Not Have Matches to Either Multi-task or CLPsych
deduped_labels = deduped_labels.loc[(deduped_labels.user_id_str.isin(multitask_nn_labels.user_id_str) | deduped_labels.user_id_str.isin(clp_df.user_id_str))]

## Check Label Alignment
deduped_disorders = deduped_labels.set_index("user_id_str")[disorders].copy()
multitask_disorders = multitask_nn_labels.set_index("user_id_str")[all_conditions].copy()
clpsych_disorders = clp_df.set_index("user_id_str")[["depression","ptsd","neurotypical"]].copy()
merged_disorders = pd.concat([multitask_disorders, clpsych_disorders], sort=True).fillna(0).astype(int)

## Merge Data Sets
merged_df = pd.concat([clp_df, multitask_nn_labels],
                      sort=True)

## Format Conditions
for d in all_conditions:
    merged_df[d] = merged_df[d].fillna(0).astype(int)

## Add Dataset Indicators
merged_df["clpsych"] = merged_df.user_id_str.isin(clpsych_disorders.index)
merged_df["clpsych_deduped"] = merged_df.clpsych & merged_df.user_id_str.isin(deduped_disorders.index)
merged_df["multitask"] = merged_df.user_id_str.isin(multitask_disorders.index)
merged_df["merged"] = merged_df.user_id_str.isin(deduped_disorders.index)

## Sort Columns
col_order = ["user_id_str", "source", "age", "gender", "split", "match"] + all_conditions + ["clpsych","clpsych_deduped","multitask","merged"]
merged_df = merged_df[col_order].copy()

## Rename Columns and Re-index
merged_df = merged_df.rename(columns={"match":"user_id_str_matched"})
merged_df = merged_df.reset_index(drop=True).copy()

## Format Matched User ID Str
merged_df["user_id_str_matched"] = merged_df["user_id_str_matched"].fillna("None")

###################
### Cache
###################

## Choose Output File
outfile = f"{QNTFY_PATH}qntfy_merged_labels.csv"
merged_df.to_csv(outfile, index=False)

