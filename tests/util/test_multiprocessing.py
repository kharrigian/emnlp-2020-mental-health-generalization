
##########################
### Imports
##########################

## External Libraries
import pytest

## Local
from mhlib.util.multiprocessing import MyPool

##########################
### Globals
##########################

NUM_PROCESSES = 4

##########################
### Helpers
##########################

def a(l1):
    p1 = MyPool(NUM_PROCESSES)
    res = p1.map(b, l1)
    p1.close()
    return res

def b(l2):
    p2 = MyPool(NUM_PROCESSES)
    res = p2.map(c, l2)
    p2.close()
    return res

def c(x):
    return x**2

##########################
### Tests
##########################

def test_MyPool():
    """
    """
    ## Sample Data
    l1 = [list(range(10)) for _ in range(20)]
    ## Sample Multiprocessing Results
    l1_mp_res = a(l1)
    ## Sample Serial Result
    l1_ser_res = []
    for _ in range(20):
        l1_ser_res.append(list(map(lambda x: c(x), list(range(10)))))
    assert l1_mp_res == l1_ser_res

