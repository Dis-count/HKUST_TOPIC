import gurobipy as grb
from gurobipy import GRB
import numpy as np
from collections import Counter
from SamplingMethodSto import samplingmethod1
from Mist import generate_sequence, decision1
import time

# This function uses IP to solve stochastic Model directly.
class originalModel:
    def __init__(self, roll_width, given_lines, demand_width_array, num_sample, I, prop, dw, s):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.s = s
        self.value_array = demand_width_array - self.s
        self.W = len(prop)
        self.I = I
        self.dw = dw
        self.prop = prop
        self.num_sample = num_sample

    def Wmatrix(self):
        # n is the dimension of group types
        return - np.identity(self.I) + np.eye(self.I, k=1)

    def solveModelGurobi(self):
        # self.prop = self.prop * self.num_sample
        m2 = grb.Model()
        x = m2.addVars(self.I, self.given_lines, lb=0,
                       vtype = GRB.INTEGER, name='varx')
        y1 = m2.addVars(self.I, self.W, lb=0,  vtype = GRB.INTEGER)
        y2 = m2.addVars(self.I, self.W, lb=0,  vtype = GRB.INTEGER)
        W0 = self.Wmatrix()
        m2.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                   for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        M_identity = np.identity(self.I)

        m2.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) + grb.quicksum(W0[i, j] * y1[j, w] + M_identity[i, j]*y2[j, w] for j in range(self.I)) == self.dw[w][i] for i in range(self.I) for w in range(self.W))
        # print("Constructing second took...", round(time.time() - start, 2), "seconds")
        m2.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(
            self.given_lines)) - grb.quicksum(y1[i, w]*self.prop[w] for i in range(self.I) for w in range(self.W)), GRB.MAXIMIZE)

        m2.setParam('OutputFlag', 0)
        # m2.Params.MIPGapAbs = 1
        m2.Params.MIPGap = 1e-6
        # m2.setParam('TimeLimit', 120)
        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        solx = sol[0: self.I * self.given_lines]
        # soly2 = sol[-self.I * self.W:]
        # print(f'check the result:{any(soly2)}')
        newx = np.reshape(solx, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)
        print('obj:', m2.objVal)
        return newd, newx

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    number_period = 60
    given_lines = 8
    np.random.seed(0)
    sd = 1
    probab = [0.4, 0.4, 0.1, 0.1]
    sam = samplingmethod1(I, num_sample, number_period, probab, sd)

    # roll_width = np.arange(21, 21 + given_lines)
    roll_width = np.ones(given_lines) * 21

    demand_width_array = np.arange(2, 2+I)

    sequence = generate_sequence(number_period, probab, sd)
    dw, prop = sam.get_prob_ini(sequence[0])
    W = len(dw)
    
    my = originalModel(roll_width, given_lines,
                         demand_width_array, num_sample, I, prop, dw, sd)

    start = time.time()
    ini_demand, upperbound = my.solveModelGurobi()
    print(ini_demand)
    print("LP took...", round(time.time() - start, 3), "seconds")