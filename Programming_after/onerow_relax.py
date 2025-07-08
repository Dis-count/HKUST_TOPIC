import gurobipy as grb
from gurobipy import GRB
import numpy as np
from collections import Counter
from SamplingMethodSto import samplingmethod1
from Mist import generate_sequence, decision1
import time

# This function uses onerow_relaxation(N rows constraints relaxed to one row) to solve stochastic Model directly.

class relaxModel:
    def __init__(self, roll_width, given_lines, demand_width_array, num_sample, I, prop, dw, s):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.all_width = sum(self.roll_width)
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
        x = m2.addVars(self.I, lb=0,
                       vtype=GRB.CONTINUOUS, name='varx')
        y1 = m2.addVars(self.I, self.W, lb=0,  vtype= GRB.CONTINUOUS)
        y2 = m2.addVars(self.I, self.W, lb=0,  vtype= GRB.CONTINUOUS)
        W0 = self.Wmatrix()
        m2.addConstr(grb.quicksum(self.demand_width_array[i] * x[i] for i in range(self.I)) <= self.all_width)
        M_identity = np.identity(self.I)

        m2.addConstrs(x[i] + grb.quicksum(W0[i, j] * y1[j, w] +
                      M_identity[i, j]*y2[j, w] for j in range(self.I)) == self.dw[w][i] for i in range(self.I) for w in range(self.W))
        # print("Constructing second took...", round(time.time() - start, 2), "seconds")
        m2.setObjective(grb.quicksum(self.value_array[i] * x[i] for i in range(self.I)) - grb.quicksum(y1[i, w]*self.prop[w] for i in range(self.I) for w in range(self.W)), GRB.MAXIMIZE)

        m2.setParam('OutputFlag', 0)
        # m2.Params.MIPGapAbs = 1
        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        solx = sol[0:self.I]
        # soly2 = sol[-self.I * self.W:]
        # print(f'check the result:{any(soly2)}')
        newd = np.sum(solx)
        print(solx)
        print(m2.ObjVal)
        return newd, solx


if __name__ == "__main__":
    num_sample = 10000  # the number of scenarios
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

    my = relaxModel(roll_width, given_lines,
                       demand_width_array, num_sample, I, prop, dw, sd)

    start = time.time()
    ini_demand, upperbound = my.solveModelGurobi()
    print("LP took...", round(time.time() - start, 3), "seconds")
