import gurobipy as grb
from gurobipy import GRB
import numpy as np
from Method10 import deterministicModel
from Method8 import column_generation
from SamplingMethodSto import samplingmethod1
import copy

# improved bid-price

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, num_period, value, weight, demand_array):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.value_array = value
        self.demand_array = demand_array
        self.weight = weight
        self.I = I   # number, Immutable object
        self.num_period = num_period   # number, Immutable object   
    # Used to generate the sequence with the first one fixed.

    def bid_price(self, sequence):
        # Original bid-price control policy.
        # Don;t solve LP.
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)

        for t in range(self.num_period):
            i = sequence[t]
            
            if max(roll_width) < i:
                decision_list[t] = 0
            elif np.isin(self.weight[i], roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == self.weight[i])[0]
                roll_width[j[0]] -= self.weight[i]

            else:
                demand = (self.num_period - t) * np.array(self.probab)
                # demand[i-1-self.s] += 1
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
                demand_capa = demand * self.weight
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
            demand[i] += 1
        test = deterministicModel(self.roll_width, self.given_lines, self.weight, self.I, self.value)
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
            # demand[i-1] += 1

            improved = column_generation(roll_width, self.given_lines, self.weight, self.I, self.value)

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
            i = sequence[t]

            if np.isin(self.weight[i], roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == self.weight[i])[0]
                roll_width[j[0]] -= self.weight[i]
                continue
            demand = (self.num_period - t) * np.array(self.probab)
            # demand[i-1] += 1

            improved = column_generation(roll_width, self.given_lines, self.weight, self.I, self.value)

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
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)
        sequence = [i-self.s for i in sequence]

        for t in range(self.num_period):
            i = sequence[t]

            if max(roll_width) < self.weight[i]:
                decision_list[t] = 0
            elif np.isin(self.weight[i], roll_width).any():
                decision_list[t] = 1
                j = np.where(roll_width == self.weight[i])[0]
                roll_width[j[0]] -= self.weight[i]
            else:
                demand = (self.num_period - t) * np.array(self.probab)
                # demand[i-1] += 1

                improved = column_generation(roll_width, self.given_lines, self.weight, self.I, self.value)

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

def generate_sequence(period, prob):
    I = len(prob)
    item_type = np.arange(I)
    trials = [np.random.choice(item_type, p = prob) for _ in range(period)]
    return trials

if  __name__ == "__main__":
    given_lines = 10

    roll_width = np.ones(given_lines) * 21
    # roll_width = np.ones(given_lines) * 12
    # roll_width = np.array([16,17,18,19,20,21,22, 23, 23,24])
    # 0.12, 0.5, 0.13, 0.25
    # 0.34, 0.51, 0.07, 0.08
    num_period = 60
    value = np.array([4, 6, 8])
    weight = np.array([3, 4, 5])
    demand_array = np.array([2, 4, 10])
    I = 4
    multi = np.arange(1, I+1)
    probab = np.array([0.12, 0.5, 0.13, 0.25])
    
    a = CompareMethods(roll_width, given_lines, I, num_period, value, weight, demand_array)

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