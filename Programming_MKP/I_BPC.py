import gurobipy as grb
from gurobipy import GRB
import numpy as np
from Method10 import deterministicModel
from Method8 import column_generation
import copy

# improved bid-price

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, num_period, value, weight, probab):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.value_array = value
        self.weight = weight
        self.I = I   # number, Immutable object
        self.num_period = num_period   # number, Immutable object   
        self.probab = probab
    # Used to generate the sequence with the first one fixed.

    def bid_price(self, sequence):
        # Original bid-price control policy.
        # Don;t solve LP.
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)

        for t in range(self.num_period):
            i = sequence[t]
            demand = (self.num_period - t) * np.array(self.probab)
            if max(roll_width) < self.weight[i]:
                decision_list[t] = 0
            elif np.isin(self.weight[i], roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == self.weight[i])[0]
                roll_width[j[0]] -= self.weight[i]

            else:

                demand_capa = demand * self.weight
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
        # BPC
        # Don;t solve LP. With simple break tie.
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)
        roll_length = sum(self.roll_width)

        for t in range(self.num_period):
            i = sequence[t]-1
            demand = (self.num_period - t-1) * np.array(self.probab)
            demand[i] += 1

            if max(roll_width) < self.weight[i]:
                decision_list[t] = 0
            elif np.isin(self.weight[i], roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == self.weight[i])[0]
                roll_width[j[0]] -= self.weight[i]
                roll_length -= self.weight[i]
            else:

                demand_capa = demand * self.weight
                demand_capa = np.cumsum(demand_capa[::-1])

                length_index = self.I-1
                for index, length in enumerate(demand_capa):
                    if length >= roll_length:
                        length_index = index
                        break

                if i + 1 >= self.I - length_index and max(roll_width) >= self.weight[i]:
                    decision_list[t] = 1

                    for j in range(self.given_lines):
                        if roll_width[j] >= self.weight[i]:
                            roll_width[j] -= self.weight[i]
                            roll_length -= self.weight[i]
                            break

                else:
                    decision_list[t] = 0


        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel(self.roll_width, self.given_lines, self.weight, self.I, self.value_array)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)

        return newd

    def improved_origin(self, sequence):
        #  max{x_ij}
        # break tie - min
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)

        for t in range(self.num_period):
            i = sequence[t]

            if np.isin(self.weight[i], roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == self.weight[i])[0]
                roll_width[j[0]] -= self.weight[i]
                continue
            # stationary demand rate
            demand = (self.num_period - t) * np.array(self.probab)


            improved = column_generation(roll_width, self.given_lines, self.weight, self.I, self.value_array)

            dom_set = [np.zeros((1, self.I)) for _ in range(self.given_lines)]

            opt_x, _ = improved.setGeneration(dom_set, demand, roll_width)
            # print(f'arrival:{i}')
            # print(f'xij: {opt_x[i-1]}')
            # print(roll_width)
            j = max(opt_x[i])
            # j_index = np.argmin(opt_x[i-1])
            arr = opt_x[i]
            if j > 0:
                decision_list[t] = 1
                j_index = np.argmin(np.where(arr > 0, arr, np.inf))
                roll_width[j_index] -= self.weight[i-1]
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

        for t in range(self.num_period):
            i = sequence[t]-1
            if np.isin(self.weight[i], roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == self.weight[i])[0]
                roll_width[j[0]] -= self.weight[i]
                continue
            demand = (self.num_period - t- 1) * np.array(self.probab)
            demand[i] += 1
            improved = column_generation(roll_width, self.given_lines, self.weight, self.I, self.value_array)

            dom_set = [np.zeros((1, self.I)) for _ in range(self.given_lines)]

            opt_x, _ = improved.setGeneration(dom_set, demand, roll_width)
            # print(f'arrival:{i}')
            # print(f'xij: {opt_x[i-1]}')
            # print(roll_width)
            j = max(opt_x[i])
            j_index = np.argmax(opt_x[i])

            if j > 0:
                decision_list[t] = 1
                roll_width[j_index] -= self.weight[i]
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
        # BPP
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)

        for t in range(self.num_period):
            i = sequence[t]-1
            demand = (self.num_period -t-1) * np.array(self.probab)
            demand[i] += 1

            if max(roll_width) < self.weight[i]:
                decision_list[t] = 0
            elif np.isin(self.weight[i], roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == self.weight[i])[0]
                roll_width[j[0]] -= self.weight[i]
            else:
                
                improved = column_generation(roll_width, self.given_lines, self.weight, self.I, self.value_array)

                dom_set = [np.zeros((1, self.I)) for _ in range(self.given_lines)]
                alpha, beta, gamma, dom_set = improved.setGeneration_bid(dom_set, demand, roll_width)

                # find sum_{\beta h} = \gamma
                find_j = -1

                for line_idx in range(self.given_lines):
                    if find_j > -1:
                        break
                    for pattern in dom_set[line_idx]:
                        if pattern[i] >=1 and np.dot(beta[:, line_idx], pattern) == gamma[line_idx]:
                            find_j = line_idx
                            break

                if find_j >= 0:
                    decision_list[t] = 1
                    roll_width[find_j] -= self.weight[i]
                else:
                    decision_list[t] = 0

        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

def generate_sequence(period, prob):
    #  generate item type from 1 
    I = len(prob)
    item_type = np.arange(1,I+1)
    trials = [np.random.choice(item_type, p = prob) for _ in range(period)]
    return trials

if  __name__ == "__main__":
    given_lines = 4

    # roll_width = np.ones(given_lines) * 21
    # roll_width = np.ones(given_lines) * 12
    roll_width = np.array([6,7,8,9])

    num_period = 10
    value = np.array([4, 6, 8])
    weight = np.array([3, 4, 5])
    # demand_array = np.array([2, 4, 10])
    I = 3

    probab = np.array([0.2, 0.5, 0.3])
    # demand_array = num_period * probab

    a = CompareMethods(roll_width, given_lines, I,
                       num_period, value, weight, probab)

    # sequence = generate_sequence(num_period, probab)
    sequence = [2, 1, 2, 2, 2, 2, 2, 3, 1, 2]
    # print(sequence)
    c = a.bid_price_1(sequence)
    print(f'BPC: {np.dot(value, c)}')

    b = a.improved_bid(sequence)
    print(f'BPP: {np.dot(value, b)}')
    print(f'BPP_demand: {b}')
    
    d = a.dynamic_primal(sequence)
    print(f'primal_demand: {d}')
    print(f'primal: {np.dot(value, d)}')

    f = a.offline(sequence)  # optimal result
    print(f'optimal_demand:{f}')

    optimal = np.dot(value, f)
    print(f'optimal: {optimal}')

    # newx = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 1.0], [-0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    # change_roll = np.array([0,0,0,0,0,0,0,5,13,4])
    # new = a.full_largest(newx.T, change_roll)
    # print(new)