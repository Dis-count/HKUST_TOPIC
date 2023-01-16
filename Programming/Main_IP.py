# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method4 import deterministicModel
from Method2 import originalModel
from Mist import generate_sequence, decision1
import copy
from Mist import decisionSeveral, decisionOnce
import time
# This function call different methods


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

        m2 = originalModel(self.roll_width, self.given_lines, self.demand_width_array, self.num_sample, self.I, prop, dw)

        ini_demand2 = m2.solveModelGurobi()

        deter = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)

        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand)
        ini_demand, _ = deter.IP_formulation(ini_demand, np.zeros(self.I))


        return sequence, ini_demand, ini_demand2

    def method1(self, sequence, ini_demand):

        decision_list = decision1(sequence, ini_demand, self.probab)
        sequence = [i-1 for i in sequence if i > 0]

        final_demand = np.array(sequence) * np.array(decision_list)
        # print('The result of Method 1--------------')
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand

    def result(self, sequence, ini_demand, ini_demand2):

        final_demand1 = self.method1(sequence, ini_demand)

        final_demand2 = self.method1(sequence, ini_demand2)

        return final_demand1, final_demand2


if __name__ == "__main__":

    num_sample = 5000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 300
    given_lines = 30
    # np.random.seed(i)
    probab = [0.25, 0.25, 0.25, 0.25]
    # probab = [0.3, 0.5, 0.1, 0.1]

    roll_width = np.ones(given_lines) * 41
    # total_seat = np.sum(roll_width)

    a_instance = CompareMethods(
        roll_width, given_lines, I, probab, num_period, num_sample)

    ratio1 = 0
    ratio2 = 0

    accept_people = 0
    num_people = 0

    multi = np.arange(1, I+1)

    count = 50
    for j in range(count):
        sequence, ini_demand, ini_demand2 = a_instance.random_generate()

        a, b = a_instance.result(sequence, ini_demand, ini_demand2)

        # ratio1 += (np.dot(multi, a)-baseline)/ baseline

        ratio1 += np.dot(multi, a)
        ratio2 += np.dot(multi, b)

    print('%.2f' % (ratio1/count))
    print('%.2f' % (ratio2/count))

