from collections import Counter
from scipy.stats import binom
import numpy as np
import copy

def generate_sequence(period, prob):
    I = len(prob)
    item_type = np.arange(1, 1+ I)
    trials = [np.random.choice(item_type, p = prob) for _ in range(period)]
    return trials

def sequence_pool(count, num_period, probab):
    pools = np.zeros((count, num_period), dtype = int)
    for i in range(count):
        pools[i] = generate_sequence(num_period, probab)
    return pools

