# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method2 import originalModel
from Method4 import deterministicModel
from Mist import generate_sequence, decision1
from collections import Counter
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
        sam = samplingmethod(self.I, self.num_sample, self.num_period, self.probab)

        dw, prop = sam.get_prob()
        W = len(dw)

        sequence = generate_sequence(self.num_period, self.probab)

        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw)

        m2 = originalModel(self.roll_width, self.given_lines,
                             self.demand_width_array, self.num_sample, self.I, prop, dw)
        
        ini_demand2, _ = m2.solveModelGurobi()

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)


        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I)

        ini_demand2, _ = deter.IP_formulation(np.zeros(self.I), ini_demand2) 
        ini_demand2, _ = deter.IP_formulation(ini_demand2, np.zeros(self.I))

        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand) 
        ini_demand, _ = deter.IP_formulation(ini_demand, np.zeros(self.I))


        ini_demand1 = np.array(self.probab) * self.num_period

        ini_demand3, _ = deter.IP_formulation(np.zeros(self.I), ini_demand1)
        ini_demand3, _ = deter.IP_formulation(ini_demand3, np.zeros(self.I))

        return sequence, ini_demand, ini_demand2, ini_demand3

    def row_by_row(self, sequence):
        # i is the i-th request in the sequence
        # j is the j-th row
        # sequence includes social distance.
        remaining_capacity = np.zeros(self.given_lines)
        current_capacity = copy.deepcopy(self.roll_width)
        j = 0
        period  = 0
        for i in sequence:
            if i in remaining_capacity:
                inx = np.where(remaining_capacity == i)[0][0]
                remaining_capacity[inx] = 0

            if current_capacity[j] > i:
                current_capacity[j] -= i
            else:    
                remaining_capacity[j] = current_capacity[j]
                j +=1
                if j > self.given_lines-1:
                    break
                current_capacity[j] -= i
            period +=1
        
        lis = [0] * (self.num_period - period)
        for k, i in enumerate(sequence[period:]):
            if i in remaining_capacity:
                inx = np.where(remaining_capacity == i)[0][0]
                remaining_capacity[inx] = 0
                lis[k] = 1
        my_list111 = [1]* period + lis
        sequence = [i-1 for i in sequence]

        final_demand = np.array(sequence) * np.array(my_list111)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand

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
                if i == remaining:
                    seq = sequence[0:res] + [i]
        return seq

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        sequence = [i-1 for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)
        newd, obj = test.IP_formulation(np.zeros(self.I), demand)

        return newd

    def method4(self, sequence, ini_demand):
        mylist = []
        remaining_period0 = self.num_period
        sequence1 = copy.copy(sequence)
        total_usedDemand = np.zeros(self.I)
        # ini_demand1 = np.array(self.probab) * self.num_period
        deterModel = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)

        while remaining_period0:
            demand = ini_demand - total_usedDemand
            usedDemand, remaining_period = decisionSeveral(sequence, demand)
            diff_period = remaining_period0 - remaining_period

            mylist += [1] * diff_period

            if any(usedDemand) == 0:  # all are 0
                usedDemand, decision_list = decisionOnce(sequence, demand, self.probab)
                if decision_list:
                    mylist.append(1)
                else:
                    mylist.append(0)
                remaining_period -= 1

            remaining_period0 = remaining_period
            sequence = sequence[-remaining_period:]

            total_usedDemand += usedDemand

            ini_demand, obj = deterModel.IP_formulation(
                total_usedDemand, np.zeros(self.I))

        sequence1 = [i-1 for i in sequence1 if i > 0]
        # total_people1 = np.dot(sequence1, mylist)

        final_demand1 = np.array(sequence1) * np.array(mylist)
        final_demand1 = final_demand1[final_demand1 != 0]

        demand = np.zeros(self.I)
        for i in final_demand1:
            demand[i-1] += 1
        
        return demand

    def method1(self, sequence, ini_demand):

        decision_list = decision1(sequence, ini_demand, self.probab)
        sequence = [i-1 for i in sequence if i > 0]

        final_demand = np.array(sequence) * np.array(decision_list)
        # print('The result of Method 1--------------')
        final_demand = final_demand[final_demand!=0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand

    def result(self, sequence, ini_demand, ini_demand2, ini_demand3):
        ini_demand4 = copy.deepcopy(ini_demand)

        final_demand1 = self.method1(sequence, ini_demand)

        final_demand2 = self.method1(sequence, ini_demand2)

        final_demand3 = self.method4(sequence, ini_demand3)

        final_demand4 = self.method4(sequence, ini_demand4)


        return final_demand1, final_demand2, final_demand3, final_demand4


if __name__ == "__main__":

    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 37
    given_lines = 8
    # np.random.seed(i)
    probab = [0.25, 0.25, 0.25, 0.25]

    roll_width = np.ones(given_lines) * 20

    total_seat = np.sum(roll_width)

    a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)

    ratio1 = 0
    ratio2 = 0
    ratio3 = 0
    ratio4 = 0
    ratio5 = 0
    ratio6 = 0

    multi = np.arange(1, I+1)

    count = 50
    for j in range(count):
        sequence, ini_demand, ini_demand2, ini_demand3 = a_instance.random_generate()

        a,b,c,d = a_instance.result(sequence, ini_demand, ini_demand2, ini_demand3)
        
        e = a_instance.row_by_row(sequence)

        baseline = np.dot(multi, e)

        f = a_instance.offline(sequence)

        seq = a_instance.binary_search_first(sequence)

        g = a_instance.offline(seq)

        ratio1 += (np.dot(multi, a)-baseline)/ baseline
        ratio2 += (np.dot(multi, b)-baseline) / baseline
        ratio3 += (np.dot(multi, c)-baseline) / baseline
        ratio4 += (np.dot(multi, d)-baseline) / baseline
        ratio5 += (np.dot(multi, f)-baseline) / baseline
        ratio6 += (np.dot(multi, g)-baseline) / baseline

    print("%.2f" % (ratio1/count*100))
    print('%.2f' % (ratio2/count*100))
    print('%.2f' % (ratio3/count*100))
    print('%.2f' % (ratio4/count*100))
    print('%.2f' % (ratio5/count*100))
    print('%.2f' % (ratio6/count*100))
