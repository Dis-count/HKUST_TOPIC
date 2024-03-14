import numpy as np
from SamplingMethodSto import samplingmethod1
from Method1 import stochasticModel
from Method10 import deterministicModel
from Mist import generate_sequence, decision1
import copy
from Mist import decisionOnce
from typing import List
from Method_scenario import stochasticModel1
import time

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
        first_arrival = sequence[0]
        dw, prop = sam.get_prob_ini(first_arrival)
        W = len(dw)
        m1 = stochasticModel(self.roll_width, self.given_lines, self.demand_width_array, W, self.I, prop, dw, self.s)

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)

        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)

        ini_demand, newx4 = deter.IP_formulation(np.zeros(self.I), ini_demand)
        newx4 = self.full_largest(newx4, self.roll_width)

        return sequence, ini_demand, newx4

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
        threshold = sum(self.roll_width) - self.given_lines*(self.I + self.s-1)
        current = 0
        acc_demand = np.zeros(self.I)
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
        sequence = [i-self.s for i in sequence if i > 0]

        for k, i in enumerate(sequence):  # i = 1,2,3,4
            decision = record[S][T][i-1]
            if not decision:
                decision_list[k] = 0
            elif decision and current <= threshold:
                S -= i+1
                decision_list[k] = 1
                acc_demand[i-1] += 1
                current += i+1
            else:
                deter1 = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
                acc_demand[i-1] += 1
                indi = deter1.IP_formulation1(acc_demand, np.zeros(self.I))
                if indi:
                    decision_list[k] = 1
                    S -= i+1
                else:
                    decision_list[k] = 0
                    acc_demand[i-1] -= 1
            T -= 1

        return acc_demand

    def bid_price(self, sequence):
        threshold = sum(self.roll_width) - self.given_lines*(self.I + self.s-1)
        current = 0
        acc_demand = np.zeros(self.I)
        decision_list = [0] * self.num_period
        roll_width = sum(self.roll_width)

        for t in range(self.num_period):
            i = sequence[t]

            demand = (self.num_period - t) * np.array(self.probab)

            deterModel = deterministicModel(roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
            value, obj = deterModel.LP_formulation_relax(demand, roll_width)
            decision = (i-self.s) - value * i

            if decision < 0:
                decision_list[t] = 0
            elif decision >=0 and current <= threshold:
                decision_list[t] = 1
                acc_demand[i-self.s-1] += 1
                current += i
                roll_width -= i
            else:
                deter1 = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
                acc_demand[i-self.s-1] += 1
                indi = deter1.IP_formulation1(acc_demand, np.zeros(self.I))
                if indi:
                    decision_list[t] = 1
                    roll_width -= i
                else:
                    decision_list[t] = 0
                    acc_demand[i-self.s-1] -= 1
        return acc_demand

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        sequence = [i-self.s for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)
        return newd

    def rearrange(self, change_roll, remaining_period, seq):
        sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
        dw_without, prop_without = sam_multi.accept_sample(seq)
        W_without = len(dw_without)
        m_without = stochasticModel(change_roll, self.given_lines,
                                self.demand_width_array, W_without, self.I, prop_without, dw_without, self.s)
        ini_demand_without, _ = m_without.solveBenders(eps=1e-4, maxit=20)
        deterModel = deterministicModel(change_roll, self.given_lines, self.demand_width_array, self.I, self.s)
        _, newx = deterModel.IP_formulation(np.zeros(self.I), ini_demand_without)

        newx = self.full_largest(newx, change_roll)

        newx = newx.T.tolist()
        return newx

    def full_largest(self, newx, change_roll):
        for new_num, new_i in enumerate(newx.T):
            occu = np.dot(new_i, self.demand_width_array)
            delta = int(change_roll[new_num] - occu)

            while delta > 0:
                for d_num, d_i in enumerate(new_i[0:-1]):
                    if d_i > 0:
                        k1 = max(1, self.I- d_num-1)
                        new_i[d_num] -= 1
                        k2 = min(d_num+1+delta, self.I)-1
                        new_i[k2] += 1
                        delta -= k1
                        break
                while delta >= self.I + self.s:
                    delta -= self.I + self.s
                    new_i[self.I-1] += 1

                if delta > self.s:
                    new_i[delta-self.s-1] += 1
                    delta = 0
                else:
                    delta = 0
        return newx

    def break_tie(self, newx, change_roll, group_type):
        newx = np.array(newx).T
        a = np.argwhere(newx[group_type] > 1e-4)
        length = np.dot(np.arange(self.s+1, self.I+self.s+1), newx)
        beta = change_roll - length
        b = beta[a]
        a_min = a[np.argmin(b)][0]   # find the minimum index in a
        return a_min

    def break_tie2(self, newx, change_roll, group_type):
        newx = np.array(newx).T
        a = np.argwhere(newx[group_type] > 1e-4)
        length = np.dot(np.arange(self.s+1, self.I+self.s+1), newx)
        beta = change_roll - length
        b = beta[a]
        a_max = a[np.argmax(b)][0]   # find the maximum index in a
        return a_max

    def method_new(self, sequence: List[int], newx, change_roll0):
        change_roll = copy.deepcopy(change_roll0)
        # threshold = sum(self.roll_width) - self.given_lines*(self.I + self.s-1)
        # current = 0
        # acc_demand = np.zeros(self.I)
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
                if j == self.s + self.I and newd[-1] == 0:
                    newx = self.rearrange(change_roll, remaining_period, j)
                    newd = np.sum(newx, axis=0)
            else:
                usedDemand, decision_list = decisionOnce(sequence[-remaining_period:], newd, self.probab, self.s)
                Indi_Demand = np.dot(usedDemand, range(self.I))

                change_deny = copy.deepcopy(change_roll)
                if decision_list:
                    k = self.break_tie2(newx, change_roll, decision_list)
                    newx[k][decision_list] -= 1
                    if decision_list - Indi_Demand - 1 - self.s >= 0:
                        newx[k][int(decision_list - Indi_Demand - 1 - self.s)] += 1
                    change_roll[k] -= (Indi_Demand + 1 + self.s)

                    change_accept = copy.deepcopy(change_roll)
                    sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
                    dw_acc, prop_acc = sam_multi.get_prob()
                    W_acc = len(dw_acc)
                    m_acc = stochasticModel(change_accept, self.given_lines,
                                            self.demand_width_array, W_acc, self.I, prop_acc, dw_acc, self.s)
                    ini_demand_acc, val_acc = m_acc.solveBenders(eps=1e-4, maxit=20)

                    dw_deny = dw_acc
                    prop_deny = prop_acc
                    W_deny = len(dw_deny)
                    m_deny = stochasticModel(change_deny, self.given_lines,
                                             self.demand_width_array, W_deny, self.I, prop_deny, dw_deny, self.s)
                    ini_demand_deny, val_deny = m_deny.solveBenders(eps=1e-4, maxit=20)
                    # ini_demand_deny = np.ceil(ini_demand_deny)

                    if val_acc + (j-self.s) < val_deny:
                        mylist.append(0)
                        deterModel = deterministicModel(change_deny, self.given_lines, self.demand_width_array, self.I, self.s)
                        _, newx = deterModel.IP_formulation(np.zeros(self.I), ini_demand_deny)

                        newx = self.full_largest(newx, change_deny)
                        newx = newx.T.tolist()
                        change_roll = change_deny
                    else:
                        mylist.append(1)
                        deterModel = deterministicModel(change_accept, self.given_lines, self.demand_width_array, self.I, self.s)
                        _, newx = deterModel.IP_formulation(np.zeros(self.I), ini_demand_acc)

                        newx = self.full_largest(newx, change_accept)
                        newx = newx.T.tolist()
                        change_roll = change_accept
                else:
                    mylist.append(0)

        sequence = [i-self.s for i in sequence]
        final_demand = np.array(sequence) * np.array(mylist)
        final_demand = final_demand[final_demand != 0]
        print(change_roll)
        print(mylist)
        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

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
    # np.random.seed(3)
    roll_width = np.ones(given_lines) * 21
    # roll_width = np.array([210])
    num_period = 70
    I = 4
    probab = np.array([0.25, 0.25, 0.25, 0.25])
    num_sample = 1000
    s = 1
    a = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, s)
    sequence, ini_demand, newx4 = a.random_generate()

    new = a.bid_price(sequence)
    
    test = a.dynamic_program(sequence)

    optimal = a.offline(sequence)

    multi = np.arange(1,1+I)
    new_value = np.dot(multi, new)
    test_value = np.dot(multi, test)
    opt_value = np.dot(multi, optimal)

    print(f'bid: {new_value}')
    print(f'DP: {test_value}')
    print(f'optimal: {opt_value}')

