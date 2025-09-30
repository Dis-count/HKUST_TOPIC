import gurobipy as grb
from gurobipy import GRB
import numpy as np
from Method10 import deterministicModel
from Method8 import column_generation
from SamplingMethodSto import samplingmethod1
import copy

# improved bid-price

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, s):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.s = s
        self.value_array = np.arange(1, 1+I)
        self.demand_width_array = self.value_array + s
        self.I = I   # number, Immutable object
        self.probab = probab
        self.num_period = num_period   # number, Immutable object   
    # Used to generate the sequence with the first one fixed.

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
        # Original bid-price control policy.
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
                # demand[i-1-self.s] += 1
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

    def bid_price_1(self, sequence):
        # Original bid-price control policy.
        # Don;t solve LP. With simple break tie.
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
                # demand[i-1-self.s] += 1
                demand_capa = demand * np.arange(1+self.s, 1+self.s+self.I)
                demand_capa = np.cumsum(demand_capa[::-1])

                length_index = self.I-1
                for index, length in enumerate(demand_capa):
                    if length >= roll_length:
                        length_index = index
                        break

                if i - self.s >= self.I - length_index and max(roll_width) >= i:
                    decision_list[t] = 1

                    for j in range(self.given_lines):
                        if roll_width[j] >= i:
                            roll_width[j] -= i
                            roll_length -= i
                            break

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
        for i in sequence:
            demand[i-1-self.s] += 1
        test = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)

        return newd

    def improved_origin(self, sequence):
        #  max{x_ij}
        # break tie - min
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)
        sequence = [i-self.s for i in sequence]

        for t in range(self.num_period):
            i = sequence[t]

            if np.isin(i+self.s, roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == i+self.s)[0]
                roll_width[j[0]] -= i+self.s
                continue
            demand = (self.num_period - t) * np.array(self.probab)
            # demand[i-1] += 1

            improved = column_generation(roll_width, self.given_lines, self.demand_width_array, self.I, self.s)

            dom_set = [np.zeros((1, self.I)) for _ in range(self.given_lines)]

            opt_x, _ = improved.setGeneration(dom_set, demand, roll_width)
            # print(f'arrival:{i}')
            # print(f'xij: {opt_x[i-1]}')
            # print(roll_width)
            j = max(opt_x[i-1])
            # j_index = np.argmin(opt_x[i-1])
            arr = opt_x[i-1]
            if j > 0:
                decision_list[t] = 1
                j_index = np.argmin(np.where(arr > 0, arr, np.inf))
                roll_width[j_index] -= self.demand_width_array[i-1]
            else:
                decision_list[t] = 0
            
        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

    def dynamic_primal(self, sequence):
        #  max{x_ij}
        # break_tie - MAX
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)
        sequence = [i-self.s for i in sequence]

        for t in range(self.num_period):
            i = sequence[t]

            if np.isin(i+self.s, roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == i+self.s)[0]
                roll_width[j[0]] -= i+self.s
                continue
            demand = (self.num_period - t) * np.array(self.probab)
            # demand[i-1] += 1

            improved = column_generation(
                roll_width, self.given_lines, self.demand_width_array, self.I, self.s)

            dom_set = [np.zeros((1, self.I)) for _ in range(self.given_lines)]

            opt_x, _ = improved.setGeneration(dom_set, demand, roll_width)
            # print(f'arrival:{i}')
            # print(f'xij: {opt_x[i-1]}')
            # print(roll_width)
            j = max(opt_x[i-1])
            j_index = np.argmax(opt_x[i-1])

            if j > 0:
                decision_list[t] = 1
                roll_width[j_index] -= self.demand_width_array[i-1]
            else:
                decision_list[t] = 0

        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

    def improved_bid(self, sequence):
        #  max{i - beta_ij}
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)
        sequence = [i-self.s for i in sequence]

        for t in range(self.num_period):
            i = sequence[t]

            if max(roll_width) < i+self.s:
                decision_list[t] = 0
            elif np.isin(i+self.s, roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == i+self.s)[0]
                roll_width[j[0]] -= i+self.s
            else:
                demand = (self.num_period - t) * np.array(self.probab)
                # demand[i-1] += 1

                improved = column_generation(roll_width, self.given_lines, self.demand_width_array, self.I, self.s)

                dom_set = [np.zeros((1, self.I)) for _ in range(self.given_lines)]
                beta = improved.setGeneration_bid(dom_set, demand, roll_width)

                delta = i - beta[i-1]
                indices = [ind for ind, val in enumerate(roll_width) if val >= i + self.s]
                b_values = [delta[i] for i in indices]

                j = max(b_values)
                j_index = indices[np.argmax(b_values)]

                print(f'arrival:{i}')
                print(f'value: {delta}')
                print(f'index: {j_index}')
                
                if j > 0:
                    decision_list[t] = 1
                    roll_width[j_index] -= i+ self.s
                else:
                    decision_list[t] = 0
                print(roll_width)
        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

def generate_sequence(period, prob, sd):
    I = len(prob)
    group_type = np.arange(1 + sd, 1 + sd + I)
    trials = [np.random.choice(group_type, p=prob) for _ in range(period)]
    return trials

if  __name__ == "__main__":
    given_lines = 10

    roll_width = np.ones(given_lines) * 21
    # roll_width = np.ones(given_lines) * 12
    # roll_width = np.array([16,17,18,19,20,21,22, 23, 23,24])
    # 0.12, 0.5, 0.13, 0.25
    # 0.34, 0.51, 0.07, 0.08
    num_period = 60
    I = 4
    multi = np.arange(1, I+1)
    probab = np.array([0.12, 0.5, 0.13, 0.25])
    s = 1
    a = CompareMethods(roll_width, given_lines, I, probab, num_period, s)

    # sequence = generate_sequence(num_period, probab, s)

    # sequence = [3, 3, 2, 3, 5, 2, 5, 3, 2, 2, 3, 2, 3, 2, 2, 5, 3, 3, 2, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 2, 5, 2, 2, 2, 2, 2, 2, 5, 2, 3, 3, 3, 5, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 4, 4, 2, 5, 3, 3, 3, 2, 5, 5, 5, 3, 3, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 2, 5, 5] total 90

    sequence =  [3, 3, 5, 2, 3, 5, 5, 3, 4, 3, 5, 2, 3, 3, 5, 2, 3, 2, 5, 3, 5, 4, 5, 5, 3, 5, 2, 2, 4, 3, 4, 5, 4, 3, 4, 4, 3, 3, 3, 3, 3, 5, 4, 2, 4, 3, 3, 3, 3, 3, 3, 3, 2, 5, 4, 5, 5, 5, 5, 5]

    sequence = sequence[:60]

    b = a.improved_bid(sequence)
    print(f'improved_bid: {np.dot(multi, b)}')

    # c = a.bid_price(sequence)
    # print(f'bid-price: {np.dot(multi, c)}')

    f = a.offline(sequence)  # optimal result
    print(f)
    optimal = np.dot(multi, f)
    print(f'optimal: {optimal}')


    # newx = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 1.0], [-0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    # change_roll = np.array([0,0,0,0,0,0,0,5,13,4])
    # new = a.full_largest(newx.T, change_roll)
    # print(new)