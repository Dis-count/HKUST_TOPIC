#  test the gap between LP and IP

import gurobipy as grb
from gurobipy import GRB
import numpy as np


class deterministicModel:
    def __init__(self, roll_width, given_lines, demand_width_array, I, s):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array - s
        self.I = I
        self.s = s

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
        aa = np.arange(1+self.s, 1+self.s+self.I)
        remaining = self.roll_width - np.dot(newx.T, aa)
        return newd, remaining

    def LP_formulation(self, demand):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb = 0, vtype = GRB.CONTINUOUS)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        m.addConstrs(grb.quicksum(x[i, j] for j in range(
                self.given_lines)) <= demand[i] for i in range(self.I))
        
        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(
            self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()

        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)
        # m.objVal
        remaining = sum(self.roll_width) - np.dot(newd, np.arange(1+self.s, 1+self.s+self.I))
        return newd, remaining



if __name__ == "__main__":
    for i in range(10000):
        s = 1
        I = 4
        demand_width_array = np.arange(1+s, 1+s+I)
        
        given_lines = 6
        roll_width = np.random.randint(5, 20, size=given_lines)
        # roll_width = np.array([7,7,7,7,7])
        test = deterministicModel(roll_width, given_lines,
                                demand_width_array, I, s)
        demand = np.random.randint(0, 10, size=I)
        demand[0] = given_lines
        demand[1] = max(1, demand[1])
        demand[2] = max(1, demand[2])
        # demand = np.array([13,13,13,13])
        d1, val1 = test.IP_formulation(demand)
        # d2, val2 = test.LP_formulation(demand)
        # A = -(val2 - val1)
        # B = given_lines * s
        # print(val1)
        d_gap = demand - d1
        if  max(val1)>1.5 and sum(d_gap) > 0:
            print(val1)
            print(demand_width_array)
            print(roll_width)
            print(demand)
            print(d1)
            break