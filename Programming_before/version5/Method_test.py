import gurobipy as grb
from gurobipy import GRB
import numpy as np
import random

# This function uses deterministicModel to make several decisions with initial deterministic solution.

class deterministicModel:
    def __init__(self, roll_width, given_lines, demand_width_array, I, s):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.s = s
        self.value_array = demand_width_array - s
        self.I = I

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
        m.optimize()
        # print('************************************************')
        # print('Optimal value of IP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)
        return newd, m.objVal

    def IP_advanced(self, demand_lower):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype = GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))

        for k in range(self.I):
            m.addConstr(grb.quicksum(x[i, j] for j in range(
                self.given_lines) for i in range(self.I-k-1, self.I, 1)) >= grb.quicksum(demand_lower[i] for i in range(self.I-k-1, self.I, 1)))

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

if  __name__ == "__main__":
    I = 4  # the number of group types
    given_lines = 10
    sd = 1

    # roll_width = np.arange(21, 21 + given_lines)
    roll_width = np.ones(given_lines) * 21
    
    demand_width_array = np.arange(2, 2+I)

    deterModel = deterministicModel(roll_width, given_lines, demand_width_array, I, sd)

    number_period = random.randint(60, 89)
    probab = [0.3, 0.2, 0.3, 0.2]

    demand_upper = np.array(probab) * number_period
    total_usedDemand, _ = deterModel.IP_formulation(demand_upper)
    print(total_usedDemand)
    ini_demand, obj = deterModel.IP_advanced(total_usedDemand)

    print(ini_demand)
    multi = np.arange(1, 1+I)
    new_value = np.dot(multi, ini_demand)

