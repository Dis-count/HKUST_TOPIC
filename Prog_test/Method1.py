import gurobipy as grb
from gurobipy import GRB
import numpy as np
import time
import math
from SamplingMethodSto import samplingmethod1
from Mist import generate_sequence
# This function solves stochastic Model. Including IP and LP.

class stochasticModel:
    def __init__(self, roll_width, given_lines, demand_width_array, W, I, prop, dw, s):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.s = s
        self.value_array = demand_width_array - self.s
        self.W = W
        self.I = I
        self.dw = dw
        self.prop = prop

    def IP_formulation(self, demand_upper):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype=GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))

        m.addConstrs(grb.quicksum(x[i, j] for j in range(
                self.given_lines)) <= demand_upper[i] for i in range(self.I))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(
            self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        # m.setParam('TimeLimit', 40)
        m.optimize()

        # print('Optimal value of IP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newx = np.rint(newx)
        newd = np.sum(newx, axis=1)
        newd = np.rint(newd)
        return newd, newx

    def obtainY(self, ind_dw, d0):
        yplus = np.zeros(self.I)
        yminus = np.zeros(self.I)

        for j in range(self.I-1, -1, -1):
            if ind_dw[j] > d0[j]:
                yminus[j] = ind_dw[j] - d0[j]
            else:
                yplus[j] = - ind_dw[j] + d0[j]
        return yplus, yminus

    def value(self, d0, d_w):
        # This funtion is used to calculate objective value when we know the optimal solution 
        yplus, yminus = self.obtainY(d_w, d0)
        T = np.zeros(self.I)
        Z = np.zeros(self.I)
        Z[-1] = (self.I+1) * yplus[-1]
        T[-1] = - yminus[-1]

        for i in range(self.I-2, -1, -1):
            T[i] = yplus[i] - yminus[i] + math.floor(Z[i+1]/(i+1+self.s))
            Z[i] = max(T[i], 0) * (i+1+self.s)

        obj = sum((i+1) * min(T[i], 0) for i in range(self.I)) + sum((i+1) * d_w[i] for i in range(self.I))
        print('Z:', Z)
        return obj

    def total(self, d0):
        final_obj = 0
        for w in range(self.W):
            final_obj += self.value(d0, self.dw[w])
        return final_obj/self.W

    def Wmatrix(self):
        # n is the dimension of group types
        return - np.identity(self.I) + np.eye(self.I, k=1)

    def solveModelGurobi(self):
        # self.prop = self.prop * self.num_sample
        m2 = grb.Model()
        x = m2.addVars(self.I, self.given_lines, lb=0,
                       vtype=GRB.INTEGER, name='varx')
        y1 = m2.addVars(self.I, self.W, lb=0,  vtype=GRB.INTEGER)
        y2 = m2.addVars(self.I, self.W, lb=0,  vtype=GRB.INTEGER)
        W0 = self.Wmatrix()
        m2.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                   for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        M_identity = np.identity(self.I)

        m2.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) + grb.quicksum(W0[i, j] * y1[j, w] +
                      M_identity[i, j]*y2[j, w] for j in range(self.I)) == self.dw[w][i] for i in range(self.I) for w in range(self.W))
        # print("Constructing second took...", round(time.time() - start, 2), "seconds")
        m2.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(
            self.given_lines)) - grb.quicksum(y1[i, w]*self.prop[w] for i in range(self.I) for w in range(self.W)), GRB.MAXIMIZE)

        m2.setParam('OutputFlag', 0)

        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        solx = sol[0: self.I * self.given_lines]
        # soly2 = sol[-self.I * self.W:]
        # print(f'check the result:{any(soly2)}')
        newx = np.reshape(solx, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)
        # print('obj:', m2.objVal)
        return newd, m2.objVal


if __name__ == "__main__":
    num_sample = 10  # the number of scenarios
    I = 4  # the number of group types
    number_period = 60
    given_lines = 8
    np.random.seed(0)
    sd = 1
    probab = [0.4, 0.4, 0.1, 0.1]
    sam = samplingmethod1(I, num_sample, number_period, probab, sd)

    roll_width = np.ones(given_lines) * 14
    demand_width_array = np.arange(2, 2+I)
    demand_width_array = np.arange(2, 2+I)

    sequence = generate_sequence(number_period, probab, sd)
    dw, prop = sam.get_prob_ini(sequence[0])
    W = len(dw)
    my = stochasticModel(roll_width, given_lines, demand_width_array, W, I, prop, dw, sd)

    expected_d = [number_period * i for i in probab]

    d0, _ = my.IP_formulation(expected_d)
    # print(d0)
    d0 = [0, 0, 8, 16]
    obj = my.total(d0)

    print(obj)
    ini_demand, upperbound = my.solveModelGurobi()
    print(ini_demand)
    print(upperbound)