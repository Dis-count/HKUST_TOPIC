# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method4 import deterministicModel
from Mist import generate_sequence, decision1
import copy
import time
import matplotlib.pyplot as plt

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
        print(demand)
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


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 200
    given_lines = 10
    # np.random.seed(i)
    p = [[0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.85, 0.05]]
    p_len = len(p)
    c_value = np.zeros(p_len)
    people_value = np.zeros(p_len)

    begin_time = time.time()
    filename = 'different_c' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

        # c = x1 + 2 * x2 + 3 * x3 + 4* x4 
    cnt = 0
    for probab in p:

        my_file.write('probabilities: \t' + str(probab) + '\n')
        # probab = [0.3, 0.5, 0.1, 0.1]

        roll_width = np.ones(given_lines) * 21
        # total_seat = np.sum(roll_width)

        a_instance = CompareMethods(
            roll_width, given_lines, I, probab, num_period, num_sample)

        ratio6 = 0
        accept_people = 0
        # num_people = 0

        multi = np.arange(1, I+1)

        c_p = multi @ probab
        c_value[cnt] = c_p
        
        count = 1
        for j in range(count):
            sequence = a_instance.random_generate()
            # total_people = sum(sequence) - num_period

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            seq = a_instance.binary_search_first(sequence)
            g = a_instance.offline(seq)

            ratio6 += np.dot(multi, g)
            accept_people += optimal
            # num_people += total_people

        my_file.write('M6: %.2f \n' % (ratio6/count))
        my_file.write('Number of accepted optimal people: %.2f \t' %
                      (accept_people/count))

        people_value[cnt] = ratio6/count
        cnt += 1
    print(sum(roll_width) * (c_value[1]/(c_value[1]+1)))
    
    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t %f \n' % run_time)

    my_file.close()
