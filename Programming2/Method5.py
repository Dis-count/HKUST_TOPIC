import gurobipy as grb
from gurobipy import GRB
import numpy as np
import copy
from SamplingMethodNew import samplingmethod1
from Mist import generate_sequence, decision1
from Method1 import stochasticModel

# This function solves deterministicModel without social distancing.

class deterministicModel1:
    def __init__(self, roll_width, given_lines, demand_width_array, I):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.s = 0
        self.value_array = demand_width_array - self.s
        self.I = I
        # self.value_array = np.array([i*(1-0.001*(self.I - i)) for i in self.demand_width_array])

    def IP_formulation(self, demand_lower, demand_upper):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb = 0, vtype = GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        if sum(demand_upper)!= 0:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) <= demand_upper[i] for i in range(self.I))
        if sum(demand_lower)!= 0:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(
            self.given_lines)) >= demand_lower[i] for i in range(self.I))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(
            self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()

        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newx = np.rint(newx)
        newd = np.sum(newx, axis=1)
        return newd, newx

    def IP_formulation1(self, demand_lower, demand_upper):
        # This function is used to check whether the model have the optimal solution.
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

        if m.status == GRB.OPTIMAL:
            return True
        else:
            return False

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    number_period = 200
    given_lines = 10
    # np.random.seed(0)

    probab = [0.3, 0.2, 0.2, 0.3]
    sam = samplingmethod1(I, num_sample, number_period, probab)

    dw, prop = sam.get_prob()
    W = len(dw)

    roll_width = np.ones(given_lines) * 21
    demand_width_array = np.arange(1, 1+I)

    sequence = generate_sequence(number_period, probab)
    ini_demand1 = np.array(probab) * number_period

    my = stochasticModel(roll_width, given_lines,
                         demand_width_array, W, I, prop, dw)

    ini_demand, upperbound = my.solveBenders(eps=1e-4, maxit=20)

    deterModel = deterministicModel1(roll_width, given_lines, demand_width_array, I)
    ini_demand, _ = deterModel.IP_formulation(np.zeros(I), ini_demand)
    deterModel.IP_formulation(ini_demand, np.zeros(I))
