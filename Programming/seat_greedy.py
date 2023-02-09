# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method4 import deterministicModel
from Method_dynamic import dynamicWay
from Mist import generate_sequence, decision1
import copy
from Mist import decisionSeveral, decisionOnce
import time

# This function call different methods(Use method_dynamic)


class CompareMethods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, num_sample):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.demand_width_array = np.arange(2, 2+I)
        self.value_array = self.demand_width_array - 1
        self.I = I   # number, Immutable object
        self.probab = probab
        self.num_period = num_period   # number, Immutable object
        self.num_sample = num_sample   # number, Immutable object

    def random_generate(self):
        sam = samplingmethod(self.I, self.num_sample,
                             self.num_period, self.probab)

        dw, prop = sam.get_prob()
        W = len(dw)

        sequence = generate_sequence(self.num_period, self.probab)

        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw)

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)

        deter = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)

        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand)
        ini_demand, _ = deter.IP_formulation(ini_demand, np.zeros(self.I))

        return sequence, ini_demand

    def row_by_row(self, sequence):
        # i is the i-th request in the sequence
        # j is the j-th row
        # sequence includes social distance.
        remaining_capacity = np.zeros(self.given_lines)
        current_capacity = copy.deepcopy(self.roll_width)
        j = 0
        period = 0
        for i in sequence:
            if i in remaining_capacity:
                inx = np.where(remaining_capacity == i)[0][0]
                remaining_capacity[inx] = 0

            if current_capacity[j] > i:
                current_capacity[j] -= i
            else:
                remaining_capacity[j] = current_capacity[j]
                j += 1
                if j > self.given_lines-1:
                    break
                current_capacity[j] -= i
            period += 1

        lis = [0] * (self.num_period - period)
        for k, i in enumerate(sequence[period:]):
            if i in remaining_capacity:
                inx = np.where(remaining_capacity == i)[0][0]
                remaining_capacity[inx] = 0
                print(inx)
                lis[k] = 1
        my_list111 = [1] * period + lis
        sequence = [i-1 for i in sequence]

        final_demand = np.array(sequence) * np.array(my_list111)
        final_demand = final_demand[final_demand != 0]

        print(remaining_capacity)
        print(current_capacity)
        print(f'seq: {sequence}')
        print(f'lis: {my_list111}')

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand




if __name__ == "__main__":

    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 80
    given_lines = 10
    np.random.seed(0)
    probab = [0.25, 0.25, 0.25, 0.25]


    roll_width = np.ones(given_lines) * 21
    # total_seat = np.sum(roll_width)

    a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)
    ratio5 = 0

    accept_people = 0
    num_people = 0

    multi = np.arange(1, I+1)

    count = 1
    for j in range(count):
        sequence, ini_demand = a_instance.random_generate()

        e = a_instance.row_by_row(sequence)
        baseline = np.dot(multi, e)


