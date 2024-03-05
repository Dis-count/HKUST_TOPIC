import gurobipy as grb
from gurobipy import GRB
import numpy as np
from SamplingMethodSto import samplingmethod1
from collections import Counter
import copy
from typing import List
from Method1 import stochasticModel

def method_scenario(self, sequence: List[int], change_roll0):
    change_roll = copy.deepcopy(change_roll0)
    mylist = []
    periods = len(sequence)
    
    for num, j in enumerate(sequence):
        remaining_period = periods - num

        sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
        dw_acc, prop_acc = sam_multi.get_prob()
        W_acc = len(dw_acc)
        m = stochasticModel(change_roll, self.given_lines, self.demand_width_array, W_acc, self.I, prop_acc, dw_acc, self.s)

        _, xk = m.solveBenders_LP(j, eps=1e-4, maxit=20)

        if sum(xk) < 1e-4:
            mylist.append(0)
        else:
            k = np.nonzero(xk)
            change_roll[k[0][0]] -= j
            mylist.append(1)

    sequence = [i-self.s for i in sequence]
    final_demand = np.array(sequence) * np.array(mylist)
    final_demand = final_demand[final_demand != 0]

    demand = np.zeros(self.I)
    for i in final_demand:
        demand[i-1] += 1
    print(change_roll)
    print(mylist)
    return demand