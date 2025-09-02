import numpy as np
from SamplingMethodSto import samplingmethod1
from Method1 import stochasticModel
from Method10 import deterministicModel
import copy
from Mist import decisionOnce
from typing import List

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, num_sample, s):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.s = s
        self.value_array = np.arange(1, 1+I)
        self.demand_width_array = self.value_array + s
        self.I = I   # number, Immutable object
        self.probab = probab
        self.num_period = num_period   # number, Immutable object
        self.num_sample = num_sample   # number, Immutable object        
    # Used to generate the sequence with the first one fixed.
    def random_generate(self, sequence):
        sam = samplingmethod1(self.I, self.num_sample, self.num_period-1, self.probab, self.s)
        first_arrival = sequence[0]
        dw, prop = sam.get_prob_ini(first_arrival)
        m1 = stochasticModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, prop, dw, self.s)

        ini_demand, _ = m1.solveBenders(eps = 1e-4, maxit = 20)
        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
        ini_demand, newx4 = deter.IP_formulation(np.zeros(self.I), ini_demand)
        _, newx4 = deter.IP_advanced(ini_demand)

        return newx4

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
        #  value is used to store the DP values
        decision_list = [0] * T

        for k, i in enumerate(sequence):
            for lines in range(self.given_lines):
                if roll_width_dy1[lines] == self.s:
                    roll_width_dy1[lines] = 0
                    S -= self.s

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
        # print(decision_list)
        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]
        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1-self.s] += 1

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
        # print(f'bid: {roll_width}')
        return demand

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        for i in sequence:
            demand[i-1-self.s] += 1
        test = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)
        return newd

    def rearrange(self, change_roll, remaining_period, seq):
        sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
        dw_without, prop_without = sam_multi.accept_sample(seq)
        m_without = stochasticModel(change_roll, self.given_lines,
                                self.demand_width_array, self.I, prop_without, dw_without, self.s)
        ini_demand_without, _ = m_without.solveBenders(eps = 1e-4, maxit = 20)
        deterModel = deterministicModel(change_roll, self.given_lines, self.demand_width_array, self.I, self.s)
        newd, _ = deterModel.IP_formulation(np.zeros(self.I), ini_demand_without)
        _, newx = deterModel.IP_advanced(newd)

        newx = newx.T.tolist()
        return newx

    def rearrange_1(self, change_roll, remaining_period, seq):
        # This one is used to generate the seat plan without the full or largest 
        sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
        dw_without, prop_without = sam_multi.accept_sample(seq)
        m_without = stochasticModel(change_roll, self.given_lines,
                                    self.demand_width_array, self.I, prop_without, dw_without, self.s)
        ini_demand_without, _ = m_without.solveBenders(eps=1e-4, maxit=20)
        deterModel = deterministicModel(change_roll, self.given_lines, self.demand_width_array, self.I, self.s)
        _, newx = deterModel.IP_formulation(np.zeros(self.I), ini_demand_without)

        newx = newx.T.tolist()
        return newx

    def break_tie(self, newx, change_roll, group_type):
        # (full or largest) match break tie
        newx = np.array(newx).T
        a = np.argwhere(newx[group_type] > 1e-4)
        length = np.dot(np.arange(self.s+1, self.I+self.s+1), newx)
        beta = change_roll - length
        b = beta[a]
        a_min = a[np.argmin(b)][0]   # find the minimum index in a
        return a_min

    def break_tie2(self, newx, change_roll, group_type):
        # (full or largest) does not match, break tie
        newx = np.array(newx).T
        a = np.argwhere(newx[group_type] > 1e-4)
        length = np.dot(np.arange(self.s+1, self.I+self.s+1), newx)
        beta = change_roll - length
        b = beta[a]
        a_max = a[np.argmax(b)][0]   # find the maximum index in a
        return a_max

    def break_tie3(self, newx, group_type):
        # return list including group_type
        newx = np.array(newx).T
        a = np.argwhere(newx[group_type] > 1e-4)

        return min(a)[0]

    def method_new(self, sequence: List[int], newx, change_roll0):
        # filename = 'test_spbs_' + str(probab) + '.txt'
        # my_file = open(filename, 'w')
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)
        value = self.dp1()
        for num, j in enumerate(sequence):
            # my_file.write(str(j) + '\t')

            #  Trivial situation
            if max(change_roll) < j:
                mylist.append(0)
                continue
            #  Exactly the size of j
            elif np.isin(j, change_roll).any():
                mylist.append(1)
                kk = np.where(change_roll == j)[0]
                change_roll[kk[0]] = 0
                newx[kk[0]] = np.zeros(self.I)
                continue

            newd = np.sum(newx, axis=0)
            remaining_period = periods - num

            for i in range(self.given_lines):
                if change_roll[i] == self.s:
                    change_roll[i] = 0
            sum_length = sum(change_roll)

            # if after accept j, the supply is 0, we should generate another newx.
            if j == self.s + self.I and newd[-1] == 0:
                newx = self.rearrange(change_roll, remaining_period, j)
                newd = np.sum(newx, axis=0)

            if value[int(sum_length)][remaining_period-1] - value[int(sum_length-j)][remaining_period-1] <= (j-self.s):
                #  Match the demand
                if newd[j-1-self.s] > 1e-4:
                    mylist.append(1)
                    k = self.break_tie(newx, change_roll, j-1-self.s)
                    newx[k][j-1-self.s] -= 1
                    change_roll[k] -= j
                #  Not match the demand
                else:
                    usedDemand, decision_list = decisionOnce(sequence[-remaining_period:], newd, self.probab, self.s)
                    if decision_list:
                        Indi_Demand = np.dot(usedDemand, range(self.I))
                        k = self.break_tie2(newx, change_roll, decision_list)
                        newx[k][decision_list] -= 1
                        if decision_list - Indi_Demand - 1 - self.s >= 0:
                            newx[k][int(decision_list - Indi_Demand - 1 - self.s)] += 1
                        change_roll[k] -= j
                        mylist.append(1)
                    else:
                        # print(j)
                        # print(change_roll)
                        # print(np.sum(newx, axis=0))
                        mylist.append(0)
                    #### Regenerate the planning
                    if remaining_period >= 2:
                        newx = self.rearrange(change_roll, remaining_period, sequence[num+1])
                        newd = np.sum(newx, axis=0)
            else:
                mylist.append(0)

            # my_file.write(str(mylist[-1]) + '\t')
            # my_file.write(f'capacity: {change_roll} \n')

        # my_file.close()
        sequence1 = [i-self.s for i in sequence]
        final_demand = np.array(sequence1) * np.array(mylist)
        final_demand = final_demand[final_demand != 0]
        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand

    def seatplan_without(self, sequence: List[int], newx, change_roll0):
        # filename = 'test_sp_' + str(probab) + '.txt'
        # my_file = open(filename, 'w')
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)

        for num, j in enumerate(sequence):
            # my_file.write(str(j) + '\t')
            #  Trivial situation
            if max(change_roll) < j:
                mylist.append(0)
                continue
            #  Exactly the size of j
            elif np.isin(j, change_roll).any():
                mylist.append(1)
                kk = np.where(change_roll == j)[0]
                change_roll[kk[0]] = 0
                newx[kk[0]] = np.zeros(self.I)
                continue

            newd = np.sum(newx, axis=0)
            remaining_period = periods - num

            for i in range(self.given_lines):
                if change_roll[i] == self.s:
                    change_roll[i] = 0

            # if after accept j, the supply is 0, we should generate another newx.
            if j == self.s + self.I and newd[-1] == 0:
                newx = self.rearrange_1(change_roll, remaining_period, j)
                newd = np.sum(newx, axis=0)

            #  Match the demand
            if newd[j-1-self.s] > 1e-4:
                mylist.append(1)
                k = self.break_tie3(newx, j-1-self.s)
                newx[k][j-1-self.s] -= 1
                change_roll[k] -= j
            #  Not match the demand
            else:
                usedDemand, decision_list = decisionOnce(sequence[-remaining_period:], newd, self.probab, self.s)
                if decision_list:
                    Indi_Demand = np.dot(usedDemand, range(self.I))
                    k = self.break_tie3(newx, decision_list)
                    newx[k][decision_list] -= 1
                    if decision_list - Indi_Demand - 1 - self.s >= 0:
                        newx[k][int(decision_list - Indi_Demand - 1 - self.s)] += 1
                    change_roll[k] -= j
                    mylist.append(1)
                else:
                    mylist.append(0)

                #### Regenerate the planning
                if remaining_period >= 2:
                    newx = self.rearrange_1(change_roll, remaining_period, sequence[num+1])
                    newd = np.sum(newx, axis=0)

        # my_file.close()
        sequence1 = [i-self.s for i in sequence]
        final_demand = np.array(sequence1) * np.array(mylist)
        final_demand = final_demand[final_demand != 0]
        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        # print(change_roll)
        # print(newd)

        return demand

if __name__ == "__main__":
    given_lines = 10

    roll_width = np.ones(given_lines) * 21
    # roll_width = np.ones(given_lines) * 12
    # roll_width = np.array([16,17,18,19,20,21,22, 23, 23,24])
    num_period = 90
    I = 4
    multi = np.arange(1, I+1)
    probab = np.array([0.34, 0.51, 0.07, 0.08])
    num_sample = 1000
    s = 1
    a = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, s)
    sequence = [3, 3, 2, 3, 5, 2, 5, 3, 2, 2, 3, 2, 3, 2, 2, 5, 3, 3, 2, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 2, 5, 2, 2, 2, 2, 2, 2, 5, 2, 3, 3, 3, 5, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 4, 4, 2, 5, 3, 3, 3, 2, 5, 5, 5, 3, 3, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 2, 5, 5]
    
    # [3, 3, 5, 2, 3, 3, 5, 3, 2, 3, 2, 2, 3, 3, 4, 2, 5, 5, 2, 3, 3, 4, 2, 3, 2, 2, 2, 5, 3, 5, 3, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 4, 5, 4, 2, 3, 3, 2, 3, 3, 2, 2, 2, 3, 3, 3, 4, 3, 3, 3, 3, 2, 2, 4, 2, 2, 2, 2, 3, 3, 5, 2, 3, 3, 3, 2, 3, 2, 2, 3, 2, 2, 3, 3, 2, 2, 4, 5, 3, 5]
    print(len(sequence))
    # [3, 3, 2, 3, 5, 2, 5, 3, 2, 2, 3, 2, 3, 2, 2, 5, 3, 3, 2, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 2, 5, 2, 2, 2, 2, 2, 2, 5, 2, 3, 3, 3, 5, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 4, 4, 2, 5, 3, 3, 3, 2, 5, 5, 5, 3, 3, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 2, 5, 5]
    newx4 = a.random_generate(sequence)

    b = a.method_new(sequence, newx4, roll_width)
    print(np.dot(multi, b))

    c = a.bid_price(sequence)
    print(np.dot(multi, c))

    f = a.offline(sequence)  # optimal result
    optimal = np.dot(multi, f)
    print(f'optimal: {optimal}')

    # sequence = [3, 3, 5, 2, 5, 5, 4, 5, 3, 5, 3, 3, 4, 5, 2, 5, 3, 5, 4, 2, 5, 2, 5, 5, 2, 2, 5, 5, 5, 5, 3, 3, 5, 3, 2, 5, 5, 5, 5, 2, 5, 3, 2, 3, 3, 5, 4, 5, 2, 3, 2, 4, 2, 5, 5]

    # 

    # newx = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 1.0], [-0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    # change_roll = np.array([0,0,0,0,0,0,0,5,13,4])
    # new = a.full_largest(newx.T, change_roll)
    # print(new)

