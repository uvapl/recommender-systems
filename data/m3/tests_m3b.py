import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from numpy import nan
from pandas import DataFrame, Series

    
def monotonously_decreasing(series):
    window_size = 2
    if len(series) < 2:
        return False  # We can't evaluate too small windows
    
    def average_window(series, start, window_size=window_size):
        """Returns the average of a window of size 'window_size' starting from 'start'."""
        return sum(series[start:start+window_size]) / window_size
    
    prev_avg = average_window(series, 0)
    
    for i in range(1, len(series) - window_size - 1):  # -2 since we're considering windows of 3
        curr_avg = average_window(series, i)
        if curr_avg > prev_avg:
            return False
        prev_avg = curr_avg
    
    return True

    
def test_1(costs):
    print("Testing: ", end = "")
    assert len(costs) == 100, "You didn't run the algorithm with N=100"
    assert costs[-1] < 0.02, "The final cost after running gradient descent with N=100 and alpha=0.04 should be below 0.02."
    assert monotonously_decreasing(costs), "The cost is not monotonously decreasing"
    print("success!")
    
def test_2(costs):
    print("Testing: ", end = "")
    assert costs[-1] < 0.005, "The final cost after running gradient descent should be below 0.005."
    assert monotonously_decreasing(costs), "The cost is not monotonously decreasing"
    print("success!")
    
    
def test_4(costs):
    print("Testing: ", end = "")
    assert len(costs) == 100, f"costs only has {len(costs)} values. This should be 100. Maybe you didn't run the algorithm with N=100? Or did you forget to append the cost in the second stage of the algorithm?"
    assert costs[-1] < 0.01, "The final cost after running gradient descent with N=100 and alpha=0.04 should be well below 0.01."
    assert monotonously_decreasing(costs), "The cost is not monotonously decreasing"
    print("success!")
    
    
def test_5(costs):
    print("Testing: ", end = "")
    assert costs[-1] < 0.001, "The final cost after running gradient descent should be below 0.001."
    assert monotonously_decreasing(costs), "The cost is not monotonously decreasing"
    print("success!")