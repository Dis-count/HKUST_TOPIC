import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method10 import deterministicModel
from Mist import generate_sequence, decision1, decision2
import copy
from itertools import combinations
import time

# This function use two-dimensional dynamic programming as the upper bound.

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
        sequence = generate_sequence(self.num_period, self.probab)
        i = sequence[0]
        sam = samplingmethod(self.I, self.num_sample,
                             self.num_period-1, self.probab, i)

        dw, prop = sam.get_prob()
        W = len(dw)
        # m2 = originalModel(self.roll_width, self.given_lines, self.demand_width_array, self.num_sample, self.I, prop, dw)

        # ini_demand, newx4 = m2.solveModelGurobi()

        deter = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)

        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw)

        ini_demand, ben = m1.solveBenders(eps=1e-4, maxit=20)
        ini_demand = np.ceil(ini_demand)
        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand)
        ini_demand, newx4 = deter.IP_formulation(ini_demand, np.zeros(self.I))

        ini_demand1 = np.array(self.probab) * self.num_period
        ini_demand3, _ = deter.IP_formulation(np.zeros(self.I), ini_demand1)

        ini_demand3, newx3 = deter.IP_formulation(
            ini_demand3, np.zeros(self.I))

        return sequence, ini_demand, ini_demand3, newx3, newx4

    def dynamic_program(self, sequence):
        S = int(sum(self.roll_width))
        p = self.probab
        T = self.num_period
        capa = 0  # used to indicate whether the capacity is enough
        option = self.I
        value = [[0 for _ in range(T + 1)] for _ in range(S + 1)]
        record = [[[0] * option for _ in range(T + 1)] for _ in range(S+1)]
        for i in range(1, S + 1):
            for j in range(1, T + 1):
                value[i][j] = value[i][j-1]

                everyvalue = 0
                totalvalue = 0
                for k in range(option):
                    if k == (option - 1) and (i - self.value_array[k]) >= 1:
                        everyvalue = value[i - self.value_array[k] -
                                           1][j - 1] + self.value_array[k]
                        capa = 1
                    elif (i - self.value_array[k]) >= 1:
                        everyvalue = value[i - self.value_array[k] -
                                           1][j - 1] + self.value_array[k]
                        capa = 1
                    else:
                        everyvalue = value[i][j-1]
                        capa = 0

                    if value[i][j-1] <= everyvalue and capa:  # delta_k
                        totalvalue += p[k] * everyvalue
                        record[i][j][k] = 1
                    else:
                        totalvalue += p[k] * value[i][j-1]
                value[i][j] = totalvalue

        decision_list = [0] * T
        sequence = [i-1 for i in sequence if i > 0]

        for k, i in enumerate(sequence):  # i = 1,2,3,4
            decision = record[S][T][i-1]
            if decision:
                S -= i+1
            T -= 1
            decision_list[k] = decision
        final_demand = np.array(sequence) * np.array(decision_list)

        final_demand = final_demand[final_demand != 0]
        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        deter1 = deterministicModel(self.roll_width, self.given_lines,
                                    self.demand_width_array, self.I)
        indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        while not indi:
            demand[final_demand[-1]-1] -= 1
            final_demand = final_demand[:-1]
            indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        return demand

    def dynamic_program2(self, sequence):
        S1 = int(self.roll_width[0])
        S = int(sum(self.roll_width[1:]))
        p = self.probab
        T = self.num_period

        option = self.I
        value = [[[0 for _ in range(T + 1)] for _ in range(S + 1)] for _ in range(S1 + 1)]
        record = [[[[0] * option for _ in range(T + 1)] for _ in range(S+1)] for _ in range(S1 + 1)]

        for i2 in range(1, S1+1):
            for i in range(1, S + 1):
                for j in range(1, T + 1):
                    value[i2][i][j] = value[i2][i][j-1]

                    everyvalue = value[i2][i][j-1]
                    everyvalue1 = value[i2][i][j-1]
                    everyvalue2 = value[i2][i][j-1]
                    totalvalue = 0
                    for k in range(option):
                        if (i - self.value_array[k]) >= 1:
                            everyvalue2 = value[i2][i - self.value_array[k] -
                                                    1][j - 1] + self.value_array[k]

                        if (i2 - self.value_array[k]) >= 1:
                            everyvalue1 = value[i2 - self.value_array[k] -
                                                1][i][j - 1] + self.value_array[k]
                        value_list = [everyvalue, everyvalue1, everyvalue2]
                        
                        max_value = max(value_list)
                        record[i2][i][j][k] = value_list.index(max_value)
                        
                        totalvalue += p[k] * max_value

                    value[i2][i][j] = totalvalue

        decision_list = [0] * T
        sequence = [i-1 for i in sequence if i > 0]

        for k, i in enumerate(sequence):  # i = 1,2,3,4
            decision = record[S1][S][T][i-1]
            if decision ==1:
                S -= i+1
            elif decision ==2:
                S1 -= i+1
            T -= 1
            decision_list[k] = np.sign(decision)

        return value, decision_list

    def dynamic2(self, S1, S, T):
        p = self.probab
        option = self.I
        value = [[[0 for _ in range(T + 1)] for _ in range(S + 1)]
                 for _ in range(S1 + 1)]
        record = [
            [[[0] * option for _ in range(T + 1)] for _ in range(S+1)] for _ in range(S1 + 1)]

        for i2 in range(1, S1+1):
            for i in range(1, S + 1):
                for j in range(1, T + 1):
                    value[i2][i][j] = value[i2][i][j-1]

                    everyvalue = value[i2][i][j-1]
                    everyvalue1 = value[i2][i][j-1]
                    everyvalue2 = value[i2][i][j-1]
                    totalvalue = 0
                    for k in range(option):
                        if (i - self.value_array[k]) >= 1:
                            everyvalue2 = value[i2][i - self.value_array[k] -
                                                    1][j - 1] + self.value_array[k]

                        if (i2 - self.value_array[k]) >= 1:
                            everyvalue1 = value[i2 - self.value_array[k] -
                                                1][i][j - 1] + self.value_array[k]
                        value_list = [everyvalue, everyvalue1, everyvalue2]

                        max_value = max(value_list)
                        record[i2][i][j][k] = value_list.index(max_value)

                        totalvalue += p[k] * max_value

                    value[i2][i][j] = totalvalue

        return value

    def com_dy(self, length, T):
        # Consider all combinations of each status of length.
        total = sum(length)
        min_value = 1e6
        for j in range(int(self.given_lines/2)+1):
            for i in combinations(length, j):
                sum1 = int(sum(i))
                sum2 = int(total - sum1)
                current_value = min(value[sum1][sum2][T], value[sum2][sum1][T]) 
                if min_value > current_value:
                    min_value = current_value
        return min_value

    def each_row(self, arrival, roll_width0, T):
        max_value = self.com_dy(roll_width0, T)
        index = -1
        roll_width = copy.deepcopy(roll_width0)
        for i in range(self.given_lines):
            roll_width[i] = roll_width0[i] - arrival
            if roll_width[i] >= 0:
                upper_bound = self.com_dy(roll_width, T)
                if upper_bound + arrival-1 > max_value:
                    max_value = upper_bound + arrival-1
                    index = i
            roll_width[i] = roll_width0[i]
        
        return  max_value, index

    def main_dy(self, sequence):
        decision_list = [0] * self.num_period
        cur_roll_width = copy.deepcopy(self.roll_width)
        for num, i in enumerate(sequence):
            max_value, index = self.each_row(i, cur_roll_width, self.num_period - num)
            if index >= 0:
                decision_list[num] = 1
                cur_roll_width[index] -= i
        
        sequence = [i-1 for i in sequence if i > 0]
        final_demand = np.array(sequence) * np.array(decision_list)

        final_demand = final_demand[final_demand!=0]
        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        
        return demand

    def bid_price(self, sequence):
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)

        for t in range(self.num_period):
            i = sequence[t]
            if max(roll_width) < i:
                decision_list[t] = 0
            else:
                demand = (self.num_period - t) * np.array(self.probab)

                deterModel = deterministicModel(
                    roll_width, self.given_lines, self.demand_width_array, self.I)
                value, obj = deterModel.LP_formulation(demand, roll_width)
                decision = (i-1) - value * i
                for j in range(self.given_lines):
                    if roll_width[j] < i:
                        decision[j] = -1

                val = max(decision)
                decision_ind = np.array(decision).argmax()
                if val >= 0 and roll_width[decision_ind]-i >= 0:
                    decision_list[t] = 1
                    roll_width[decision_ind] -= i
                else:
                    decision_list[t] = 0

        sequence = [i-1 for i in sequence]
        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(I)
        for i in final_demand:
            demand[i-1] += 1
        print(f'bid: {roll_width}')
        print(f'bid: {demand}')

        return demand

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


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 70
    given_lines = 10
    # np.random.seed(16)

    probab = [0.25, 0.25, 0.25, 0.25]

    roll_width = np.ones(given_lines) * 21
    # roll_width = np.array([0,0,21,21,21,21,21,21,21,21])
    # total_seat = np.sum(roll_width)

    a_instance = CompareMethods(
        roll_width, given_lines, I, probab, num_period, num_sample)

    sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()

    h = a_instance.bid_price(sequence)

    value = a_instance.dynamic2(220, 220, 71)

    a = np.array(value)
    np.save('a.npy', a)

    a = np.load('a.npy')
    value = a.tolist()

    b = a_instance.main_dy(sequence)

    multi = np.arange(1, I+1)
    print(f'dynamic: {np.dot(multi, b)}')
    print(f'bid: {np.dot(multi, h)}')
