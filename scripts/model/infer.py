
#######################
### Imports
#######################

## Standard Library
import os
import json
import gzip
import argparse
from glob import glob
from datetime import datetime

## External Library
import joblib
import pandas as pd
from tqdm import tqdm

## Local
from mhlib.util.logging import initialize_logger

#######################
### Globals
#######################

## Logging
LOGGER = initialize_logger()

#######################
### Functions
#######################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.

    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Run modeling experiments")
    ## Generic Arguments
    parser.add_argument("model",
                        type=str,
                        help="Path to cached model file (.joblib)")
    parser.add_argument("--input",
                        type=str,
                        default=None,
                        help="Path to input folder of processed *.gz files or a single processed *.gz file")
    parser.add_argument("--output_folder",
                        type=str,
                        default=None,
                        help="Name of output folder for placing predictions")
    parser.add_argument("--min_date",
                        type=str,
                        default="2000-01-01",
                        help="Lower date boundary (isoformat str) if desired")
    parser.add_argument("--max_date",
                        type=str,
                        default=None,
                        help="upper date boundary (isoformat str) if desired")
    parser.add_argument("--n_samples",
                        type=int,
                        default=None,
                        help="Number of post samples to isolate for modeling if desired")
    parser.add_argument("--randomized",
                        action="store_true",
                        default=False,
                        help="If included along with samples, will use randomized selection instead of recent")
    parser.add_argument("--drop_null",
                        action="store_true",
                        default=False,
                        help="If included, drop null rows from output")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Could not find model file {args.model}")
    if args.input is None:
        raise ValueError("Must provide --input folder or .gz file")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Could not find input filepath {args.input}")
    if args.output_folder is None:
        raise ValueError("Must provide an --output_folder argument")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    return args

def get_file_list(args):
    """

    """
    if os.path.isfile(args.input):
        return [args.input]
    elif os.path.isdir(args.input):
        return glob(f"{args.input}*.gz")
    else:
        raise ValueError("Did not recognize command line --input")

def get_date_bounds(args):
    """

    """
    min_date = pd.to_datetime(args.min_date)
    if args.max_date is None:
        max_date = pd.to_datetime(datetime.now())
    else:
        max_date = pd.to_datetime(args.max_date)
    return min_date, max_date

def main():
    """

    """
    ## Parse Command-line Arguments
    args = parse_arguments()
    ## Identify Filenames
    filenames = get_file_list(args)
    LOGGER.info("Found {} Files".format(len(filenames)))
    ## Load Model
    LOGGER.info(f"Loading Model: {args.model}")
    model = joblib.load(args.model)
    ## Get Date Boundaries
    LOGGER.info(f"Parsing Date Boundaries")
    min_date, max_date = get_date_bounds(args)
    ## Make Predictions
    y_pred = model.predict(filenames,
                           min_date=min_date,
                           max_date=max_date,
                           n_samples=args.n_samples,
                           randomized=args.randomized,
                           drop_null=args.drop_null)
    ## Cache Predictions
    pred_file = f"{args.output_folder}{model._target_disorder}.predictions.json.gz"
    LOGGER.info(f"Caching Predictions at: {pred_file}")
    with gzip.open(pred_file,"wt",encoding="utf-8") as the_file:
        json.dump(y_pred, the_file)
    LOGGER.info("Script Complete!")

#######################
### Execute
#######################

if __name__ == "__main__":
    _ = main()