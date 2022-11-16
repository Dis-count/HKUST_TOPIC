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

class compare_methods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, num_sample):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = np.arange(2, 2+I)
        self.value_array = self.demand_width_array - 1
        self.I = I
        self.probab = probab
        self.num_period = num_period
        self.num_sample = num_sample

    def random_generate(self):
        sam = samplingmethod(self.I, self.num_sample, self.num_period, self.probab)

        dw, prop = sam.get_prob()
        W = len(dw)

        sequence = generate_sequence(self.num_period, self.probab)

        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw)

        ini_demand, upperbound = m1.solveBenders(eps=1e-4, maxit=20)

        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I)

        ini_demand, obj = deter.IP_formulation(np.zeros(self.I), ini_demand)

        ini_demand1 = np.array(self.probab) * self.num_period

        ini_demand3, obj = deter.IP_formulation(np.zeros(self.I), ini_demand1)

        return sequence, ini_demand, ini_demand3


    def method4(self, sequence, ini_demand):
        mylist = []
        remaining_period0 = self.num_period
        sequence1 = copy.deepcopy(sequence)
        total_usedDemand = np.zeros(self.I)
        ini_demand1 = np.array(self.probab) * self.num_period
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

            ini_demand1 = total_usedDemand + \
                np.ceil(np.array(self.probab) * remaining_period)

            ini_demand, obj = deterModel.IP_formulation(
                total_usedDemand, ini_demand1)

        sequence1 = [i-1 for i in sequence1 if i > 0]
        total_people1 = np.dot(sequence1, mylist)
        final_demand1 = np.array(sequence1) * np.array(mylist)
        final_demand1 = final_demand1[final_demand1 != 0]
        t = Counter(final_demand1)
        demand = np.zeros(self.I)
        for k, i in enumerate(sorted(t)):
            demand[k] = t[i]
        return demand
        # print(f'The number of seats: {total_seat}')
        # print(f'The number of people:{total_people1}')
        # print(Counter(final_demand1))


    def method1(self, sequence, ini_demand):
        decision_list = decision1(sequence, ini_demand, self.probab)
        
        sequence = [i-1 for i in sequence if i > 0]
        # total_people = np.dot(sequence, decision_list)
        final_demand = np.array(sequence) * np.array(decision_list)
        # print('The result of Method 1--------------')
        # print(f'The total seats: {total_seat}')
        # print(f'The total people:{total_people}')
        # print(Counter(final_demand))
        final_demand = final_demand[final_demand!=0]
        t = Counter(final_demand)
        demand = np.zeros(self.I)
        for k,i in enumerate(sorted(t)):
            demand[k]= t[i]
        return demand

    def result(self, sequence, ini_demand, ini_demand3):
        final_demand1 = self.method1(sequence, ini_demand)
        final_demand3 = self.method4(sequence, ini_demand3)
        final_demand4 = self.method4(sequence, ini_demand)
        return final_demand1, final_demand3, final_demand4


if __name__ == "__main__":

    num_sample = 10000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 350
    given_lines = 30
    # np.random.seed(0)
    probab = [0.25, 0.25, 0.25, 0.25]

    roll_width = np.ones(given_lines) * 40

    total_seat = np.sum(roll_width)

    a_instance = compare_methods(roll_width, given_lines, I, probab, num_period, num_sample)

    final_demand1 = np.zeros(I)
    final_demand3 = np.zeros(I)
    final_demand4 = np.zeros(I)

    count = 50
    for i in range(count):
        sequence, ini_demand, ini_demand3 = a_instance.random_generate()
        a,b,c = a_instance.result(sequence, ini_demand, ini_demand3)
        final_demand1 += a
        final_demand3 += b
        final_demand4 += c

    people1 = np.dot(np.arange(1,I+1), final_demand1/count)
    people3 = np.dot(np.arange(1,I+1), final_demand3/count)
    people4 = np.dot(np.arange(1,I+1), final_demand4/count)

    # print(final_demand1/50)
    # print(final_demand3/50)
    # print(final_demand4/50)

    print(people1)
    print(people3)
    print(people4)
