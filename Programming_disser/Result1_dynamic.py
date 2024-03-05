import gurobipy as grb
from gurobipy import GRB
import numpy as np
import time
from Method2 import originalModel
from Method1 import stochasticModel
from SamplingMethodSto import samplingmethod1

#  This function is used to compare running times

for i in range(50):
    num_sample = 5000  # the number of scenarios
    number_period = 100
    I = 4  # the number of group types
    sd = 1
    given_lines = 10
    # np.random.seed(0)

    probab = [0.25, 0.25, 0.25, 0.25]
    sam = samplingmethod1(I, num_sample, number_period, probab, sd)

    dw, prop = sam.get_prob()
    W = len(dw)

    roll_width = np.ones(given_lines) * 28
    # total_seat = np.sum(roll_width)

    demand_width_array = np.arange(1+sd, 1+I+sd)

    # my = stochasticModel(roll_width, given_lines, demand_width_array, W, I, prop, dw, sd)

    my1 = originalModel(roll_width, given_lines, demand_width_array, W, I, prop, dw, sd)

    # final_d, upperbound = my.solveBenders(eps = 1e-4, maxit= 20)
    # print('ratio:', upperbound/total_seat)

    # start = time.time()
    # my.solveBenders(eps = 1e-4, maxit= 20)
    # print("Berders took...", round(time.time() - start, 2), "seconds")

    start1 = time.time()
    my1.solveModelGurobi()
    print("IP took...", round(time.time() - start1, 2), "seconds")
