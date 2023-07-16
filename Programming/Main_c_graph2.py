# import gurobipy as grb
# from gurobipy import GRB
import random
import statsmodels.api as sm
import numpy as np
from SamplingMethodNew import samplingmethod1
from Method1 import stochasticModel
from Method4 import deterministicModel
from Mist import generate_sequence, decision1
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd
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
        sam = samplingmethod1(self.I, self.num_sample,
                             self.num_period, self.probab)

        sequence = generate_sequence(self.num_period, self.probab)

        return sequence

    def binary_search_first(self, sequence):
        # Return the index not less than the first
        target = sum(self.roll_width)
        arr = np.cumsum(sequence)
        low = 0
        high = len(arr)-1
        res = -1
        while low <= high:
            mid = (low + high)//2
            if target <= arr[mid]:
                res = mid
                high = mid-1
            else:
                low = mid+1
        if res == -1:
            total = sum(sequence)
            seq = sequence
        else:
            seq = sequence[0:res]
            total = sum(seq)

        remaining = target - total
        if remaining > 0 and res > 0:
            for i in sequence[res:]:
                if i <= remaining:
                    seq = sequence[0:res] + [i]
                    remaining -= i

        seq = [i-1 for i in seq]
        demand = np.zeros(self.I)
        for i in seq:
            demand[i-1] += 1

        deter1 = deterministicModel(self.roll_width, self.given_lines,
                                    self.demand_width_array, self.I)
        indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        while not indi:
            demand[seq[-1]-1] -= 1
            seq.pop()
            indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        seq = [i+1 for i in seq]

        return seq

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        sequence = [i-1 for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)

        return newd

    def result(self, sequence, ini_demand, ini_demand3):
        ini_demand4 = copy.deepcopy(ini_demand)

        final_demand1 = self.method1(sequence, ini_demand)

        final_demand3 = self.method4(sequence, ini_demand3)

        final_demand4 = self.method4(sequence, ini_demand4)

        return final_demand1, final_demand3, final_demand4

def prop_list():
    x1 = np.arange(0.05, 1, 0.05)
    x2 = np.arange(0.05, 1, 0.05)
    x3 = np.arange(0.05, 1, 0.05)

    p = np.zeros((len(x1)*len(x2)*len(x3), 4))
    t = 0

    for i in x1:
        for j in x2:
            for k in x3:
                if 1 - i - j - k >0:
                    p[t] = [i, j, k, 1 - i - j - k]
                    t += 1

    p = p[0:t]

    return p


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 200
    given_lines = 10
    # np.random.seed(i)
    p = prop_list()
    p_len = len(p)

    c_value = np.zeros(p_len)
    people_value = np.zeros(p_len)
    optimal_value = np.zeros(p_len)
    begin_time = time.time()
    filename = 'different_c' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    count = 50

    for ind_probab, probab in enumerate(p):

        my_file.write('probabilities: \t' + str(probab) + '\n')
        # probab = [0.3, 0.5, 0.1, 0.1]

        roll_width = np.ones(given_lines) * 21
        # roll_width = np.arange(17,22)
        # roll_width= np.append(roll_width, np.arange(21,26))
        
        # roll_width = np.array([26, 21, 24, 20, 20, 17, 23, 19, 21, 19])

        # total_seat = np.sum(roll_width)
        multi = np.arange(1, I+1)

        c_p = multi @ probab
        c_value[ind_probab] = c_p

        a_instance = CompareMethods(
            roll_width, given_lines, I, probab, num_period, num_sample)

        ratio6 = 0
        accept_people = 0
        # num_people = 0

        for j in range(count):
            sequence = a_instance.random_generate()
            # total_people = sum(sequence) - num_period

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            seq = a_instance.binary_search_first(sequence)
            g = a_instance.offline(seq)

            # num_people += total_people

            ratio6 += np.dot(multi, g)
            accept_people += optimal

        optimal_value[ind_probab] = accept_people/count
        people_value[ind_probab] = ratio6/count
        my_file.write('Result: %.2f \n' % (ratio6/count))

        # my_file.write('Number of accepted people: %.2f \t' %(accept_people/count))
        # my_file.write('Number of people: %.2f \n' % (num_people/count))
        
        # if c_p < 3.2:
        #     occup_value = sum(roll_width) * (c_p/(c_p+1))
        # else:
        #     occup_value = 160
        # my_file.write('Mean estimation: %.2f \n' % occup_value)

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t %f \n' % run_time)

    occup_value = np.zeros(p_len)
    diff = np.zeros(p_len)
    ratio = 0
    for i in range(p_len):
        if c_value[i] <= 3.2:
            occup_value[i] = sum(roll_width) * (c_value[i]/(c_value[i]+1))
        else:
            occup_value[i] = 160

        diff[i] = occup_value[i]- people_value[i]

    # for i in range(p_len):
    #     if c_value[i] ==2:
            
        # if abs(diff[i]) >= 4:
        #     ratio += 1
        #     my_file.write('Deviated probability: ' + str(p[i]) + '\t')
        #     my_file.write('Deviated value: %.2f \n' % diff[i])
    
    # print(sum(abs(diff) >= 4)/p_len)
    # print(sum(abs(diff) >= 3)/p_len)
    # print(sum(abs(diff) >= 2)/p_len)
    # print(sum(abs(diff) >= 1)/p_len)

    # plt.hist(diff, bins=20, color='red', alpha=0.75)
    # plt.title('Difference Distribution')
    # plt.show()

    gamma = c_value/(c_value+1)
    plt.scatter(gamma, people_value, c = "blue")
    plt.scatter(gamma, occup_value, c = "red")
    # plt.scatter(c_value, optimal_value, c = "green")
    plt.show()

my_file.close()

