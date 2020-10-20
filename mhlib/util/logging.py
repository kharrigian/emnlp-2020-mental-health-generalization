
########################
### Imports
########################

## Standard Libarary
import sys
import logging

########################
### Functions
########################

def initialize_logger(level=logging.INFO):
    """
    Create a logger object for outputing
    to standard out

    Args:
        level (int or str): Logging filter level
    
    Returns:
        logger (Logging object): Python logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
