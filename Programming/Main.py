# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method3 import deterministicModel
from Mist import generate_sequence, decision1
from collections import Counter
import copy
from Mist import decisionSeveral, decisionOnce
# This function call different methods

# class compare_methods:
#     def __init__(self, roll_width, given_lines, demand_width_array, W, I, probab, dw, num_period):
#         self.roll_width = roll_width
#         self.given_lines = given_lines
#         self.demand_width_array = demand_width_array
#         self.value_array = demand_width_array - 1
#         self.W = W
#         self.I = I
#         self.dw = dw
#         self.probab = probab
#         self.num_period = num_period


num_sample = 1000  # the number of scenarios
I = 4  # the number of group types
number_period = 100
given_lines = 8
# np.random.seed(0)
probab = [0.4, 0.4, 0.1, 0.1]

sam = samplingmethod(I, num_sample, number_period, probab)

dw, prop = sam.get_prob()
W = len(dw)

roll_width = np.ones(given_lines) * 20
total_seat = np.sum(roll_width)

demand_width_array = np.arange(2, 2+I)

sequence = generate_sequence(number_period, probab)

m1 = stochasticModel(roll_width, given_lines,
                     demand_width_array, W, I, prop, dw)

ini_demand, upperbound = m1.solveBenders(eps=1e-4, maxit=20)

deter = deterministicModel(roll_width, given_lines, demand_width_array, I)

ini_demand, obj = deter.IP_formulation(np.zeros(I), ini_demand)

ini_demand1 = np.array(probab) * number_period

ini_demand3, obj = deter.IP_formulation(np.zeros(I), ini_demand1)

def method1(sequence, ini_demand, probab):
    decision_list = decision1(sequence, ini_demand, probab)

    sequence = [i-1 for i in sequence if i > 0]
    total_people = np.dot(sequence, decision_list)
    final_demand = np.array(sequence) * np.array(decision_list)
    # print('The result of Method 1--------------')
    # print(f'The total seats: {total_seat}')
    # print(f'The total people:{total_people}')
    # print(Counter(final_demand))
    t = Counter(final_demand)
    demand = []
    for i in sorted(t):
        demand.append(t[i])
    print(demand)
    return np.array(demand)

def method4(sequence, ini_demand, probab):
    mylist = []
    remaining_period0 = number_period
    sequence1 = copy.deepcopy(sequence)
    total_usedDemand = np.zeros(I)
    ini_demand1 = np.array(probab) * number_period
    deterModel = deterministicModel(
        roll_width, given_lines, demand_width_array, I)

    while remaining_period0:
        demand = ini_demand - total_usedDemand

        usedDemand, remaining_period = decisionSeveral(sequence, demand)

        diff_period = remaining_period0 - remaining_period

        mylist += [1] * diff_period

        if any(usedDemand) == 0:  # all are 0
            usedDemand, decision_list = decisionOnce(sequence, demand, probab)
            if decision_list:
                mylist.append(1)
            else:
                mylist.append(0)
            remaining_period -= 1

        remaining_period0 = remaining_period
        sequence = sequence[-remaining_period:]

        total_usedDemand += usedDemand

        ini_demand1 = total_usedDemand + \
            np.ceil(np.array(probab) * remaining_period)

        ini_demand, obj = deterModel.IP_formulation(
            total_usedDemand, ini_demand1)

    sequence1 = [i-1 for i in sequence1 if i > 0]
    total_people1 = np.dot(sequence1, mylist)
    final_demand1 = np.array(sequence1) * np.array(mylist)

    t = Counter(final_demand1)
    demand = []
    for i in sorted(t):
        demand.append(t[i])
    return np.array(demand)
    # print(f'The number of seats: {total_seat}')
    # print(f'The number of people:{total_people1}')
    # print(Counter(final_demand1))

def result():
    final_demand1 = method1(sequence, ini_demand, probab)
    final_demand3 = method4(sequence, ini_demand3, probab)
    final_demand4 = method4(sequence, ini_demand, probab)
    return len(probab), final_demand1, final_demand3, final_demand4


# print('The result of Method 3----------------')

# print('The result of Method 4----------------')

