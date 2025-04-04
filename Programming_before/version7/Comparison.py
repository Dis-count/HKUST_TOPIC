import numpy as np
from SamplingMethodSto import samplingmethod1
from Method1 import stochasticModel
from Method_test import deterministicModel
from Mist import generate_sequence, decision1
import copy
from Mist import decisionOnce
import time
from typing import List

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, num_sample, s):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.s = s
        self.value_array = np.arange(1, 1+I)
        self.demand_width_array = self.value_array + self.s
        self.I = I   # number, Immutable object
        self.probab = probab
        self.num_period = num_period   # number, Immutable object
        self.num_sample = num_sample   # number, Immutable object        

    def random_generate(self):
        sequence = generate_sequence(self.num_period, self.probab, self.s)
        sam = samplingmethod1(self.I, self.num_sample, self.num_period-1, self.probab, self.s)

        dw, prop = sam.get_prob_ini(sequence[0])
        W = len(dw)
        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw, self.s)

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)

        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)

        ini_demand, newx4 = deter.IP_formulation(ini_demand)
        ini_demand, newx4 = deter.IP_advanced(ini_demand)

        ini_demand1 = np.array(self.probab) * self.num_period
        ini_demand3, newx3 = deter.IP_formulation(ini_demand1)
        ini_demand3, newx3 = deter.IP_advanced(ini_demand3)
        
        return sequence, ini_demand, ini_demand3, newx3, newx4

    def row_by_row(self, sequence):
        # i is the i-th request in the sequence
        # j is the j-th row
        # sequence includes social distance.
        current_capacity = copy.deepcopy(self.roll_width)
        j = 0
        period = 0
        for i in sequence:
            if i in current_capacity:
                inx = np.where(current_capacity == i)[0][0]
                current_capacity[inx] = 0
                period += 1
                continue

            if current_capacity[j] >= i:
                current_capacity[j] -= i
            else:
                j += 1
                if j > self.given_lines-1:
                    break
                current_capacity[j] -= i
            period += 1

        lis = [0] * (self.num_period - period)
        for k, i in enumerate(sequence[period:]):
            if i in current_capacity:
                inx = np.where(current_capacity == i)[0][0]
                current_capacity[inx] = 0

                lis[k] = 1

        my_list111 = [1] * period + lis
        sequence = [i-self.s for i in sequence]

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
                if i <= remaining:
                    seq = sequence[0:res] + [i]
                    remaining -= i

        seq = [i-1 for i in seq]
        demand = np.zeros(self.I)
        for i in seq:
            demand[i-1] += 1

        deter1 = deterministicModel(self.roll_width, self.given_lines,self.demand_width_array, self.I)
        indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        while not indi:  # If indi doesnot exist, then delete the arrival from the seq.
            demand[seq[-1]-1] -= 1
            seq.pop()
            indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        seq = [i+1 for i in seq]

        return seq

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

    def dynamic_program1(self, sequence):
        roll_width_dy1 = copy.deepcopy(self.roll_width)
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
                    if k == (option - 1) and (i - self.value_array[k]) >= self.s:
                        everyvalue = value[i - self.value_array[k] -
                                           self.s][j - 1] + self.value_array[k]
                        capa = 1
                    elif (i - self.value_array[k]) >= self.s:
                        everyvalue = value[i - self.value_array[k] -
                                           self.s][j - 1] + self.value_array[k]
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
        sequence = [i-self.s for i in sequence if i > 0]

        for k, i in enumerate(sequence):  # i = 1,2,3,4
            decision = record[S][T][i-1]
            if decision:
                S -= i + self.s
                for j in range(self.given_lines):
                    if roll_width_dy1[j] >= (i+self.s):
                        roll_width_dy1[j] -= (i+self.s)
                        decision_list[k] = decision
                        break
            T -= 1

        final_demand = np.array(sequence) * np.array(decision_list)

        final_demand = final_demand[final_demand != 0]
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

                deterModel = deterministicModel(roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
                value, obj = deterModel.LP_formulation(demand, roll_width)
                decision = (i-self.s) - value * i
                for j in range(self.given_lines):
                    if roll_width[j] < i:
                        decision[j] = -1

                val = max(decision)
                decision_ind = np.array(decision).argmax()
                if val >= 0 and roll_width[decision_ind]- i >= 0:
                    decision_list[t] = 1
                    roll_width[decision_ind] -= i
                else:
                    decision_list[t] = 0

        sequence = [i-self.s for i in sequence]
        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        # print(f'bid: {roll_width}')
        return demand

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        sequence = [i-self.s for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
        newd, _ = test.IP_formulation(demand)
        return newd

    def rearrange(self, change_roll, remaining_period, seq):
        sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
        dw_without, prop_without = sam_multi.accept_sample(seq)
        W_without = len(dw_without)
        m_without = stochasticModel(change_roll, self.given_lines,
                                self.demand_width_array, W_without, self.I, prop_without, dw_without, self.s)
        ini_demand_without, _ = m_without.solveBenders(eps=1e-4, maxit=20)
        deterModel = deterministicModel(change_roll, self.given_lines, self.demand_width_array, self.I, self.s)
        newd, newx = deterModel.IP_formulation(ini_demand_without)

        _, newx = deterModel.IP_advanced(newd)

        newx = newx.T.tolist()
        return newx

    def break_tie(self, newx, change_roll, group_type):
        newx = np.array(newx).T
        a = np.argwhere(newx[group_type] > 1e-4)
        length = np.dot(np.arange(self.s+1, self.I+self.s+1), newx)
        beta = change_roll - length
        b = beta[a]
        a_min = a[np.argmin(b)][0]   # find the minimum index in a
        return a_min

    def method_new(self, sequence: List[int], newx, change_roll0):
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)
        
        for num, j in enumerate(sequence):
            newd = np.sum(newx, axis = 0)
            remaining_period = periods - num

            if newd[j-1-self.s] > 1e-4:
                mylist.append(1)
                k = self.break_tie(newx, change_roll, j-1-self.s)
                newx[k][j-1-self.s] -= 1
                change_roll[k] -= j

                newd = np.sum(newx, axis=0)
                # if after accept j, the supply is 0, we should generate another newx.
                if j == self.s + self.I and newd[-1] == 0:
                    newx = self.rearrange(change_roll, remaining_period, j)
                    newd = np.sum(newx, axis=0)
            else:
                usedDemand, decision_list = decisionOnce(sequence[-remaining_period:], newd, self.probab, self.s)
                Indi_Demand = np.dot(usedDemand, range(self.I))

                change_deny = copy.deepcopy(change_roll)
                if decision_list:
                    k = self.break_tie(newx, change_roll, decision_list)
                    newx[k][decision_list] -= 1
                    if decision_list - Indi_Demand - 1 - self.s >= 0:
                        newx[k][int(decision_list - Indi_Demand - 1 - self.s)] += 1
                    change_roll[k] -= (Indi_Demand + 1 + self.s)

                    change_accept = copy.deepcopy(change_roll)
                    sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
                    dw_acc, prop_acc = sam_multi.accept_sample(sequence[-remaining_period:][0])
                    # dw_acc, prop_acc = sam_multi.get_prob()
                    W_acc = len(dw_acc)
                    m_acc = stochasticModel(change_accept, self.given_lines,
                                            self.demand_width_array, W_acc, self.I, prop_acc, dw_acc, self.s)
                    ini_demand_acc, val_acc = m_acc.solveBenders(eps=1e-4, maxit=20)
                    # ini_demand_acc = np.ceil(ini_demand_acc)
                    dw_deny, prop_deny = sam_multi.get_prob()
                    # dw_deny = dw_acc
                    # prop_deny = prop_acc
                    W_deny = len(dw_deny)
                    m_deny = stochasticModel(change_deny, self.given_lines,
                                             self.demand_width_array, W_deny, self.I, prop_deny, dw_deny, self.s)
                    ini_demand_deny, val_deny = m_deny.solveBenders(eps=1e-4, maxit=20)
                    # ini_demand_deny = np.ceil(ini_demand_deny)

                    if val_acc + (j-self.s) < val_deny:
                        mylist.append(0)
                        deterModel = deterministicModel(change_deny, self.given_lines, self.demand_width_array, self.I, self.s)
                        newd, newx = deterModel.IP_formulation(ini_demand_deny)

                        _, newx = deterModel.IP_advanced(newd)
                        newx = newx.T.tolist()
                        change_roll = change_deny
                    else:
                        mylist.append(1)
                        deterModel = deterministicModel(change_accept, self.given_lines, self.demand_width_array, self.I, self.s)
                        newd, newx = deterModel.IP_formulation(ini_demand_acc)

                        _, newx = deterModel.IP_advanced(newd)
                        newx = newx.T.tolist()
                        change_roll = change_accept
                else:
                    mylist.append(0)

        sequence = [i-self.s for i in sequence]
        final_demand = np.array(sequence) * np.array(mylist)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

    def method_IP(self, sequence, newx, change_roll0):
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)
        for num, j in enumerate(sequence):
            newd = np.sum(newx, axis=0)
            remaining_period = periods - num
            if newd[j-1-self.s] > 0:
                mylist.append(1)
                for k, pattern in enumerate(newx):
                    if pattern[j-1-self.s] > 0:
                        newx[k][j-1-self.s] -= 1
                        change_roll[k] -= j
                        break
            else:
                mylist.append(0)
                deterModel = deterministicModel(change_roll, self.given_lines, self.demand_width_array, self.I, self.s)
                ini_demand1 = np.array(self.probab) * remaining_period
                ini_demand1 = np.ceil(ini_demand1)
                _, newx = deterModel.IP_formulation(ini_demand1)
                newx = newx.T.tolist()

        sequence = [i-self.s for i in sequence]

        final_demand = np.array(sequence) * np.array(mylist)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

    def method1(self, sequence, ini_demand):
        decision_list = decision1(sequence, ini_demand, self.probab, self.s)
        sequence = [i-1 for i in sequence if i > 0]

        final_demand = np.array(sequence) * np.array(decision_list)
        # print('The result of Method 1--------------')
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

    def result(self, sequence, ini_demand, ini_demand3, newx3, newx4):
        ini_demand4 = copy.deepcopy(ini_demand)
        ini_demand2 = copy.deepcopy(ini_demand3)
        roll_width = copy.deepcopy(self.roll_width)

        final_demand1 = self.method1(sequence, ini_demand)
        final_demand2 = self.method1(sequence, ini_demand2)

        final_demand3 = self.method_new(sequence, newx3, roll_width)
        # final_demand3 = 0
        final_demand4 = self.method_final(sequence, newx4, roll_width)

        return final_demand1, final_demand2, final_demand3, final_demand4


if __name__ == "__main__":
    given_lines = 10
    roll_width = np.ones(given_lines) * 21
    # roll_width = np.array([210])
    num_period = 70
    I = 4
    probab = np.array([0.3, 0.2, 0.1, 0.4])
    num_sample = 1000
    s = 1
    a = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, s)
    sequence, ini_demand, ini_demand3, newx3, newx4 = a.random_generate()

    # sequence = [3, 3, 5, 2, 5, 5, 4, 5, 3, 5, 3, 3, 4, 5, 2, 5, 3, 5, 4, 2, 5, 2, 5, 5, 2, 2, 5, 5, 5, 5, 3, 3, 5, 3, 2, 5, 5, 5, 5, 2, 5, 3, 2, 3, 3, 5, 4, 5, 2, 3, 2, 4, 2, 5, 5]
    # print(sum(sequence))
    new = a.method_new(sequence, newx4, roll_width)

    multi = np.arange(1,1+I)
    new_value = np.dot(multi, new)
    print(f'sto: {new_value}')

    # newx = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 1.0], [-0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    # change_roll = np.array([0,0,0,0,0,0,0,5,13,4])
    # new = a.full_largest(newx.T, change_roll)
    # print(new)