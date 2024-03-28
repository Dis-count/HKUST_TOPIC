import gurobipy as grb
from gurobipy import GRB
import numpy as np
import time
from Method2 import originalModel
from Method1 import stochasticModel

#  This function is used to compare running times

W = 10000  # the number of scenarios
I = 16  # the number of group types
sd = 1
given_lines = 30
# np.random.seed(10)
# dw = np.random.randint(20, size=(W, I)) + 20
dw = np.random.randint(low = 150, high= 250, size=(W, I))

roll_width = np.random.randint(low = 41, high = 60, size = given_lines)
# roll_width = np.random.randint(21, size = given_lines) + 30
# total_seat = np.sum(roll_width)

demand_width_array = np.arange(1+sd, 1+I+sd)
# demand_width_array = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

value_array = demand_width_array - sd

prop = np.array([1/W]*W)

my = stochasticModel(roll_width, given_lines,
                        demand_width_array, W, I, prop, dw, sd)

my1 = originalModel(roll_width, given_lines,
                    demand_width_array, W, I, prop, dw, sd)

# final_d, upperbound = my.solveBenders(eps = 1e-4, maxit= 20)
# print('ratio:', upperbound/total_seat)

start = time.time()
my.solveBenders(eps = 1e-4, maxit= 20)
print("Berders took...", round(time.time() - start, 2), "seconds")

start1 = time.time()
my1.solveModelGurobi()
print("IP took...", round(time.time() - start1, 2), "seconds")
