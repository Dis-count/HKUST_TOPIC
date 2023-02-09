# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method4 import deterministicModel
from Method5 import deterministicModel1
from Mist import generate_sequence, decision1
import copy
from Mist import decisionSeveral, decisionOnce
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
        sam = samplingmethod(self.I, self.num_sample, self.num_period, self.probab)

        dw, prop = sam.get_prob()
        W = len(dw)

        sequence = generate_sequence(self.num_period, self.probab)
        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw)

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)
        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I)

        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand)
        ini_demand, _ = deter.IP_formulation(ini_demand, np.zeros(self.I))

        return sequence, ini_demand

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        sequence = [i-1 for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel1(
            self.roll_width, self.given_lines, self.demand_width_array-1, self.I)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)
        
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

    def result(self, sequence, ini_demand):

        final_demand1 = self.method1(sequence, ini_demand)

        return final_demand1

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(30,120,1)
    given_lines = 10
    # np.random.seed(i)
    # probab = [0.25, 0.25, 0.25, 0.25]
    probab = [0.4, 0.4, 0.1, 0.1]


    t_value = np.arange(30, 120, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))

    cnt = 0
    gap_if = True
    for num_period in period_range:

        roll_width = np.ones(given_lines) * 21
        total_seat = np.sum(roll_width)

        a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)

        M1 = 0
        accept_people = 0

        multi = np.arange(1, I+1)
        print(num_period)
        count = 50
        for j in range(count):
            sequence, ini_demand = a_instance.random_generate()

            a = a_instance.result(sequence, ini_demand)
            
            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            M1 += np.dot(multi, a)
            accept_people += optimal

        occup_value[cnt] = M1/count/total_seat * 100
        people_value[cnt] = accept_people/count/total_seat * 100
        if gap_if:
            if accept_people/count - M1/count > 1:
                point = (num_period-1, occup_value[cnt-1])
                gap_if = False
        cnt += 1

    plt.plot(t_value, people_value, 'bs', label='Without social distancing')
    plt.plot(t_value, occup_value, 'r--', label='With social distancing')
    plt.xlabel('Periods')
    plt.ylabel('Percentage of total seats')
    plt.annotate('Gap', xy = point, xytext=(point[0]+10, point[1]-20), arrowprops=dict(facecolor='black', shrink=0.1),)
    plt.legend()
    plt.show()
