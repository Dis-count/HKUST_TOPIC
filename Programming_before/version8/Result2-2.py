# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from SamplingMethodSto import samplingmethod1
from Method1 import stochasticModel
from Method10 import deterministicModel
from Method2 import originalModel
from Mist import generate_sequence, decision1

# This function is used to compare the performance of seat plannings from LP and IP. 

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, num_sample, s):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.s = s
        self.demand_width_array = np.arange(1+self.s, 1+self.s+I)
        self.value_array = self.demand_width_array - self.s
        self.I = I   # number, Immutable object
        self.probab = probab
        self.num_period = num_period   # number, Immutable object
        self.num_sample = num_sample   # number, Immutable object

    def random_generate(self):
        sam = samplingmethod1(self.I, self.num_sample,
                             self.num_period, self.probab, self.s)

        sequence = generate_sequence(self.num_period, self.probab, self.s)
        dw, prop = sam.get_prob_ini(sequence[0])
        W = len(dw)

        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw, self.s)

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)

        m2 = originalModel(self.roll_width, self.given_lines, self.demand_width_array, self.num_sample, self.I, prop, dw, self.s)

        ini_demand2, _ = m2.solveModelGurobi()

        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)

        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand)
        ini_demand, _ = deter.IP_formulation(ini_demand, np.zeros(self.I))

        return sequence, ini_demand, ini_demand2

    def method1(self, sequence, ini_demand):
        decision_list = decision1(sequence, ini_demand, self.probab, self.s)
        sequence = [i-self.s for i in sequence if i > 0]

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
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 50
    given_lines = 10
    # np.random.seed(i)
    sd = 1
    probab = [0.25, 0.25, 0.25, 0.25]
    # probab = [0.3, 0.5, 0.1, 0.1]

    roll_width = np.ones(given_lines) * 21
    # total_seat = np.sum(roll_width)

    a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)

    ratio1 = 0
    ratio2 = 0

    accept_people = 0
    num_people = 0

    multi = np.arange(1, I+1)

    count = 1
    for j in range(count):
        sequence, ini_demand, ini_demand2 = a_instance.random_generate()
        print(ini_demand)
        print(ini_demand2)
        a, b = a_instance.result(sequence, ini_demand, ini_demand2)

        # ratio1 += (np.dot(multi, a)-baseline)/ baseline

        ratio1 += np.dot(multi, a)
        ratio2 += np.dot(multi, b)

    print('%.2f' % (ratio1/count))
    print('%.2f' % (ratio2/count))

