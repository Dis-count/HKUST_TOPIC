import gurobipy as grb
from gurobipy import GRB
import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method5 import deterministicModel
from Mist import generate_sequence, decision1
import copy
from Mist import decisionSeveral, decisionOnce
import time
import matplotlib.pyplot as plt
# This function call different methods


class deterministicModel:
    def __init__(self, roll_width, given_lines, demand_width_array, I):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array
        self.I = I

    def IP_formulation(self, demand_lower, demand_upper):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype=GRB.CONTINUOUS)
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
        print(f'upper: {demand_upper}')
        print(f'lower: {demand_lower}')
        m.optimize()
        print('************************************************')
        print('Optimal value of LP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)
        print(newd)
        return newd, m.objVal

if __name__ == "__main__":

    I = 4  # the number of group types

    given_lines = 10
    # np.random.seed(i)
    # probab = [0.25, 0.25, 0.25, 0.25]
    probab = [0.4, 0.4, 0.1, 0.1]

    demand_width_array = np.arange(2, 2+I)

    roll_width = np.ones(given_lines) * 21
    total_seat = np.sum(roll_width)

    a_instance = deterministicModel(roll_width, given_lines, demand_width_array, I)

    demand_lower = np.array([24, 28, 8, 9])
    demand_upper = np.zeros(I)

    a_instance.IP_formulation(demand_lower, demand_upper)
