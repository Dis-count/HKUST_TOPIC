import numpy as np
from SamplingMethodSto import samplingmethod1
from Method1 import stochasticModel
from Method10 import deterministicModel
from Mist import generate_sequence, decision1
import copy
from Mist import decisionOnce
from typing import List
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
        _, newx4 = deter.IP_advanced(ini_demand)

        return sequence, newx4

    def row_by_row(self, sequence):
        # FCFS
        # i is the i-th request in the sequence
        # j is the j-th row
        # sequence includes social distance.
        current_capacity = copy.deepcopy(self.roll_width)
        decision_list = [0] * self.num_period
        for k, i in enumerate(sequence):
            if max(current_capacity) < i:
                decision_list[k] = 0
            elif np.isin(i, current_capacity).any():
                decision_list[k] = 1
                j = np.where(current_capacity == i)[0]
                current_capacity[j[0]] -= i
            else:
                decision_list[k] = 1
                if  max(current_capacity) > self.s + self.I:
                    for j in range(self.given_lines):
                        if current_capacity[j] > self.s + self.I:
                            current_capacity[j] -= i
                            break
                else:
                    for j in range(self.given_lines):
                        if current_capacity[j] >= i:
                            current_capacity[j] -= i
                            break

        sequence = [i-self.s for i in sequence]

        final_demand = np.array(sequence) * np.array(decision_list)
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

    def dynamic_program1(self, sequence):
        roll_width_dy1 = copy.deepcopy(self.roll_width)
        S = int(sum(self.roll_width))
        p = self.probab
        T = self.num_period
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
                        everyvalue = value[i - self.value_array[k] -self.s][j - 1] + self.value_array[k]
                        capa = 1
                    elif (i - self.value_array[k]) >= self.s:
                        everyvalue = value[i - self.value_array[k] -self.s][j - 1] + self.value_array[k]
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

        for k, i in enumerate(sequence):
            if max(roll_width_dy1) < i:
                decision_list[k] = 0
            elif np.isin(i, roll_width_dy1).any():
                j = np.where(roll_width_dy1 == i)[0]
                roll_width_dy1[j[0]] -= i
                S -= i
                decision_list[k] = 1
            else:
                decision = record[S][T-k][i-1-self.s]
                if decision:
                    S -= i
                    decision_list[k] = 1
                
                    if max(roll_width_dy1) > self.s + self.I:
                        for j in range(self.given_lines):
                            if  roll_width_dy1[j] > self.s + self.I:
                                roll_width_dy1[j] -= i
                                break
                    else:
                        for j in range(self.given_lines):
                            if  roll_width_dy1[j] >= i:
                                roll_width_dy1[j] -= i
                                break
                else:
                    decision_list[k] = 0

        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]
        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1-self.s] += 1
        print(decision_list)
        print(roll_width_dy1)
        return demand

    def dp1(self):
        # Return the DP Matrix.
        S = int(sum(self.roll_width))
        p = self.probab
        T = self.num_period
        option = self.I
        value = [[0 for _ in range(T + 1)] for _ in range(S + 1)]
        for i in range(1, S + 1):
            for j in range(1, T + 1):
                value[i][j] = value[i][j-1]

                everyvalue = 0
                totalvalue = 0
                for k in range(option):
                    if k == (option - 1) and (i - self.value_array[k]) >= self.s:
                        everyvalue = value[i - self.value_array[k] -self.s][j - 1] + self.value_array[k]
                        capa = 1
                    elif (i - self.value_array[k]) >= self.s:
                        everyvalue = value[i - self.value_array[k] -self.s][j - 1] + self.value_array[k]
                        capa = 1
                    else:
                        everyvalue = value[i][j-1]
                        capa = 0

                    if value[i][j-1] <= everyvalue and capa:  # delta_k
                        totalvalue += p[k] * everyvalue
                    else:
                        totalvalue += p[k] * value[i][j-1]
                value[i][j] = totalvalue

        return value

    def bid_price(self, sequence):
        # Don;t solve LP.
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)
        roll_length = sum(self.roll_width)

        for t in range(self.num_period):
            i = sequence[t]
            
            if max(roll_width) < i:
                decision_list[t] = 0
            elif np.isin(i, roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == i)[0]
                roll_width[j[0]] -= i
                roll_length -= i
            else:
                demand = (self.num_period - t) * np.array(self.probab)

                demand_capa = demand * np.arange(1+self.s, 1+self.s+self.I)
                demand_capa = np.cumsum(demand_capa[::-1])

                length_index = self.I-1
                for index, length in enumerate(demand_capa):
                    if length >= roll_length:
                        length_index = index
                        break
                
                if max(roll_width) <= self.s + self.I:
                    tie_list = 0
                else:
                    tie_list = 1

                if  i -self.s >= self.I- length_index and tie_list:
                    decision_list[t] = 1

                    for j in range(self.given_lines):
                        if roll_width[j] > self.s + self.I:
                            roll_width[j] -= i
                            roll_length -= i
                            break

                elif i - self.s < self.I- length_index:
                    decision_list[t] = 0
                else:
                    decision_list[t] = 1
                    for j in range(self.given_lines):
                        if roll_width[j] >= i:
                            roll_width[j] -= i
                            roll_length -= i
                            break

        sequence = [i-self.s for i in sequence]
        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        # print(decision_list)
        # print(f'bid: {roll_width}')
        return demand

    def bid_price1(self, sequence):
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
                if val >= 0 and roll_width[decision_ind] - i >= 0:
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
        # print(decision_list)
        print(f'bid: {roll_width}')
        return demand

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        sequence1 = [i-self.s for i in sequence]
        for i in sequence1:
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
        newd, _ = deterModel.IP_formulation(np.zeros(self.I), ini_demand_without)
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

    def break_tie2(self, newx, change_roll, group_type):
        newx = np.array(newx).T
        a = np.argwhere(newx[group_type] > 1e-4)
        length = np.dot(np.arange(self.s+1, self.I+self.s+1), newx)
        beta = change_roll - length
        b = beta[a]
        a_max = a[np.argmax(b)][0]   # find the maximum index in a
        return a_max

    def break_tie3(self, newx, change_roll, group_type):
        # return list including group_type
        newx = np.array(newx).T
        a = np.argwhere(newx[group_type] > 1e-4)
        beta = np.arange(self.given_lines)
        return beta[a]

    def method_new(self, sequence: List[int], newx, change_roll0):
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)
        
        for num, j in enumerate(sequence):
            # print(change_roll)
            # print(newx)
            if max(change_roll) < j:
                mylist.append(0)
                continue
            elif np.isin(j, change_roll).any():
                mylist.append(1)
                kk = np.where(change_roll == j)[0]
                change_roll[kk[0]] = 0
                newx[kk[0]] = np.zeros(self.I)
                continue

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
                    k = self.break_tie2(newx, change_roll, decision_list)
                    newx[k][decision_list] -= 1
                    if decision_list - Indi_Demand - 1 - self.s >= 0:
                        newx[k][int(decision_list - Indi_Demand - 1 - self.s)] += 1
                    change_roll[k] -= (Indi_Demand + 1 + self.s)

                    change_accept = copy.deepcopy(change_roll)
                    # print(change_accept)
                    sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
                    # dw_acc, prop_acc = sam_multi.accept_sample(sequence[-remaining_period:][0])
                    dw_acc, prop_acc = sam_multi.get_prob()
                    W_acc = len(dw_acc)
                    m_acc = stochasticModel(change_accept, self.given_lines,
                                            self.demand_width_array, W_acc, self.I, prop_acc, dw_acc, self.s)
                    ini_demand_acc, val_acc = m_acc.solveBenders(eps=1e-4, maxit=20)
                    # ini_demand_acc = np.ceil(ini_demand_acc)
                    # dw_deny, prop_deny = sam_multi.get_prob()
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
                        newd, _ = deterModel.IP_formulation(np.zeros(self.I), ini_demand_deny)
                        _, newx = deterModel.IP_advanced(newd)
                        newx = newx.T.tolist()
                        change_roll = change_deny
                    else:
                        mylist.append(1)
                        deterModel = deterministicModel(change_accept, self.given_lines, self.demand_width_array, self.I, self.s)
                        newd, _ = deterModel.IP_formulation(np.zeros(self.I), ini_demand_acc)
                        _, newx = deterModel.IP_advanced(newd)
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

    def method_sto_dp(self, sequence: List[int], newx, change_roll0):
        change_roll = copy.deepcopy(change_roll0)
        L = int(sum(change_roll0))
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)
        dp = self.dp1()

        for num, j in enumerate(sequence):
            # print(self.num_period-num-1)
            # print(L)
            if max(change_roll) < j:
                mylist.append(0)
                continue
            elif np.isin(j, change_roll).any():
                mylist.append(1)
                kk = np.where(change_roll == j)[0]
                change_roll[kk[0]] = 0
                L -= j
                newx[kk[0]] = np.zeros(self.I)
                continue

            newd = np.sum(newx, axis=0)
            remaining_period = periods - num

            if num == 6:
                print(newd)
                print(newx)
                print(change_roll)

            if newd[j-1-self.s] > 1e-4:
                mylist.append(1)
                k = self.break_tie(newx, change_roll, j-1-self.s)
                newx[k][j-1-self.s] -= 1
                change_roll[k] -= j
                L -= j
                newd = np.sum(newx, axis=0)
                # if after accept j, the supply is 0, we should generate another newx.
                if j == self.s + self.I and newd[-1] == 0:
                    newx = self.rearrange(change_roll, remaining_period, j)
                    newd = np.sum(newx, axis=0)
            else:
                _, decision_list = decisionOnce(sequence[-remaining_period:], newd, self.probab, self.s)

                change_deny = copy.deepcopy(change_roll)
                if  decision_list:  
                    k = self.break_tie3(newx, change_roll, decision_list)
                    if max(change_roll[k]) > self.s + self.I:
                        for jj in k:
                            if  change_roll[jj] > self.s + self.I:
                                change_roll[jj] -= j
                                break
                    else:
                        for jj in k:
                            if  change_roll[jj] >= j:
                                change_roll[jj] -= j
                                break

                    change_accept = copy.deepcopy(change_roll)
                    sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
                    dw_acc, prop_acc = sam_multi.get_prob()
                    W_acc = len(dw_acc)
                    m_acc = stochasticModel(change_accept, self.given_lines,
                                            self.demand_width_array, W_acc, self.I, prop_acc, dw_acc, self.s)
                    ini_demand_acc, _ = m_acc.solveBenders(eps=1e-4, maxit=20)
                    dw_deny = dw_acc
                    prop_deny = prop_acc
                    W_deny = len(dw_deny)
                    m_deny = stochasticModel(change_deny, self.given_lines,self.demand_width_array, W_deny, self.I, prop_deny, dw_deny, self.s)
                    ini_demand_deny, _ = m_deny.solveBenders(eps=1e-4, maxit=20)
                    


                    if dp[L-j][self.num_period-num-1]+j-self.s < dp[L][self.num_period-num-1]:
                        mylist.append(0)
                        deterModel = deterministicModel(change_deny, self.given_lines, self.demand_width_array, self.I, self.s)
                        newd, _ = deterModel.IP_formulation(np.zeros(self.I), ini_demand_deny)
                        _, newx = deterModel.IP_advanced(newd)
                        newx = newx.T.tolist()
                        change_roll = change_deny
                    else:
                        mylist.append(1)
                        L -= j
                        deterModel = deterministicModel(change_accept, self.given_lines, self.demand_width_array, self.I, self.s)
                        newd, _ = deterModel.IP_formulation(np.zeros(self.I), ini_demand_acc)
                        _, newx = deterModel.IP_advanced(newd)
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

    def method_new_copy(self, sequence: List[int], newx, change_roll0):
        # return list
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)

        for num, j in enumerate(sequence):
            newd = np.sum(newx, axis=0)
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
                usedDemand, decision_list = decisionOnce(
                    sequence[-remaining_period:], newd, self.probab, self.s)
                Indi_Demand = np.dot(usedDemand, range(self.I))

                change_deny = copy.deepcopy(change_roll)
                if decision_list:
                    k = self.break_tie2(newx, change_roll, decision_list)
                    newx[k][decision_list] -= 1
                    if decision_list - Indi_Demand - 1 - self.s >= 0:
                        newx[k][int(decision_list - Indi_Demand -
                                    1 - self.s)] += 1
                    change_roll[k] -= (Indi_Demand + 1 + self.s)

                    change_accept = copy.deepcopy(change_roll)
                    sam_multi = samplingmethod1(
                        self.I, self.num_sample, remaining_period-1, self.probab, self.s)
                    # dw_acc, prop_acc = sam_multi.accept_sample(sequence[-remaining_period:][0])
                    dw_acc, prop_acc = sam_multi.get_prob()
                    W_acc = len(dw_acc)
                    m_acc = stochasticModel(change_accept, self.given_lines,
                                            self.demand_width_array, W_acc, self.I, prop_acc, dw_acc, self.s)
                    ini_demand_acc, val_acc = m_acc.solveBenders(
                        eps=1e-4, maxit=20)
                    # ini_demand_acc = np.ceil(ini_demand_acc)
                    # dw_deny, prop_deny = sam_multi.get_prob()
                    dw_deny = dw_acc
                    prop_deny = prop_acc
                    W_deny = len(dw_deny)
                    m_deny = stochasticModel(change_deny, self.given_lines,
                                             self.demand_width_array, W_deny, self.I, prop_deny, dw_deny, self.s)
                    ini_demand_deny, val_deny = m_deny.solveBenders(
                        eps=1e-4, maxit=20)


                    if val_acc + (j-self.s) < val_deny:
                        mylist.append(0)
                        deterModel = deterministicModel(
                            change_deny, self.given_lines, self.demand_width_array, self.I, self.s)
                        _, newx = deterModel.IP_formulation(
                            np.zeros(self.I), ini_demand_deny)

                        newx = self.full_largest(newx, change_deny)
                        newx = newx.T.tolist()
                        change_roll = change_deny
                    else:
                        mylist.append(1)
                        deterModel = deterministicModel(
                            change_accept, self.given_lines, self.demand_width_array, self.I, self.s)
                        _, newx = deterModel.IP_formulation(
                            np.zeros(self.I), ini_demand_acc)

                        newx = self.full_largest(newx, change_accept)
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
        return demand, mylist

    def method_IP(self, sequence, newx, change_roll0):
        # use the IP result to assign the groups
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)
        for num, j in enumerate(sequence):
            newd = np.sum(newx, axis=0)
            remaining_period = periods - num
            if newd[j-1-self.s] > 0:
                mylist.append(1)
                k = self.break_tie(newx, change_roll, j-1-self.s)
                newx[k][j-1-self.s] -= 1
                change_roll[k] -= j

            else:
                mylist.append(0)
                deterModel = deterministicModel(change_roll, self.given_lines, self.demand_width_array, self.I, self.s)
                ini_demand1 = np.array(self.probab) * remaining_period
                # ini_demand1 = np.ceil(ini_demand1)
                newd, _ = deterModel.IP_formulation(np.zeros(self.I), ini_demand1)
                _, newx = deterModel.IP_advanced(newd)
                newx = newx.T.tolist()

        sequence = [i-self.s for i in sequence]

        final_demand = np.array(sequence) * np.array(mylist)
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
    np.random.seed(10)
    roll_width = np.ones(given_lines) * 21
    # roll_width = np.ones(given_lines) * 12
    # roll_width = np.array([16,17,18,19,20,21,22, 23, 23,24])
    num_period = 70
    I = 4
    probab = np.array([0.25, 0.25, 0.25, 0.25])
    num_sample = 1000
    s = 1
    a = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, s)
    sequence, newx4 = a.random_generate()

    # sequence = [3, 3, 5, 2, 5, 5, 4, 5, 3, 5, 3, 3, 4, 5, 2, 5, 3, 5, 4, 2, 5, 2, 5, 5, 2, 2, 5, 5, 5, 5, 3, 3, 5, 3, 2, 5, 5, 5, 5, 2, 5, 3, 2, 3, 3, 5, 4, 5, 2, 3, 2, 4, 2, 5, 5]
    t1 = time.time()
    print(sequence)
    new = a.method_sto_dp(sequence, newx4, roll_width)
    print(new)
    # bid = a.bid_price(sequence)
    dp = a.dynamic_program1(sequence)
    # print(dp)
    # bid1 = a.bid_price1(sequence)
    # newx = copy.deepcopy(newx4)
    # optimal = a.offline(sequence)
    # print(optimal)
    multi = np.arange(1,1+I)
    new_value = np.dot(multi, new)
    # bid_value = np.dot(multi, bid)
    # dp_value = np.dot(multi, dp)
    # bid_value1 = np.dot(multi, bid1)
    # opt_value = np.dot(multi, optimal)

    print(f'sto: {new_value}')
    # print(f'bid: {bid_value}')
    # print(f'dy: {dp_value}')
    # print(f'bid1: {bid_value1}')
    # print(f'optimal: {opt_value}')

    # newx = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 1.0], [-0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    # change_roll = np.array([0,0,0,0,0,0,0,5,13,4])
    # new = a.full_largest(newx.T, change_roll)
    # print(new)