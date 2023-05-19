import gurobipy as grb
from gurobipy import GRB
import numpy as np
import time
from collections import Counter
from scipy.stats import binom
import copy


class deterministicModel:
    def __init__(self, roll_width, given_lines, demand_width_array, I):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.s = 1
        self.value_array = demand_width_array - self.s
        self.I = I
        

    def LP_formulation(self, demand, roll_width):
        m = grb.Model()
        z = m.addVars(self.I, lb=0, vtype=GRB.CONTINUOUS)
        beta = m.addVars(self.given_lines, lb=0, vtype=GRB.CONTINUOUS)

        m.addConstrs(z[i] + beta[j] * (i+1+self.s) >= i+1 for j in range(self.given_lines) for i in range(self.I))

        m.setObjective(grb.quicksum(demand[i] * z[i] for i in range(
            self.I)) + grb.quicksum(roll_width[j] * beta[j] for j in range(
                self.given_lines)), GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        # print('************************************************')
        # print('Optimal value of IP is: %g' % m.objVal)
        m.write('bid_price.lp')
        x_ij = np.array(m.getAttr('X'))[-self.given_lines:]
        return x_ij, m.objVal


class deterministicModel1:
    def __init__(self, roll_width, given_lines, demand_width_array, I):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array - 1
        self.I = I

    def IP_formulation(self, demand_lower, demand_upper):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype=GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        if sum(demand_upper) != 0:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(
                self.given_lines)) <= demand_upper[i] for i in range(self.I))
        if sum(demand_lower) != 0:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(
                self.given_lines)) >= demand_lower[i] for i in range(self.I))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(
            self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        # print('************************************************')
        # print('Optimal value of IP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)
        return newd, m.objVal

def generate_sequence(period, prob):
    trials = [np.random.choice([2, 3, 4, 5], p = prob) for i in range(period)]
    return trials

def offline(sequence):
    # This function is to obtain the optimal decision.
    demand1 = np.zeros(I)
    sequence = [i-1 for i in sequence]
    for i in sequence:
        demand1[i-1] += 1
    test = deterministicModel1(roll_width1, given_lines, demand_width_array, I)
    newd, _ = test.IP_formulation(np.zeros(I), demand1)

    return newd

if __name__ == "__main__":
    I = 4
    given_lines = 8
    roll_width = np.arange(10, 10 + given_lines)
    roll_width1 = np.arange(10, 10 + given_lines)

    probab = [0.4, 0.4, 0.1, 0.1]

    number_period = 60
    demand = number_period * np.array(probab)

    demand_width_array = np.arange(2, 2+I)

    sequence = generate_sequence(number_period, probab)
    period = len(sequence)

    decision_list = [0] * period
    
    for t in range(number_period):
        i = sequence[t]
        if max(roll_width) < i:
            decision_list[t] = 0
        else:
            demand = (number_period -t) * np.array(probab)

            deterModel = deterministicModel(roll_width, given_lines, demand_width_array, I)
            value, obj = deterModel.LP_formulation(demand, roll_width)
            decision = (i-1) - value * i
            for j in range(given_lines):
                if roll_width[j] < i:
                    decision[j] = -1

            val = max(decision)
            decision_ind = np.array(decision).argmax()
            if val >= 0 and (roll_width[decision_ind]-i) >=0:
                decision_list[t] = 1
                roll_width[decision_ind] -= i
            else:
                decision_list[t] = 0

    print(roll_width)
    print(sequence)
    multi = np.arange(1, I+1)
    f = offline(sequence)  # optimal result
    optimal = np.dot(multi, f)

    sequence = [i-1 for i in sequence]
    total_people1 = np.dot(sequence, decision_list)
    print(total_people1)
    print(decision_list)
    final_demand = np.array(sequence) * np.array(decision_list)
    final_demand = final_demand[final_demand != 0]
    
    demand = np.zeros(I)
    for i in final_demand:
        demand[i-1] += 1
    print(demand)

    print(optimal)
    print(total_people1/optimal)
