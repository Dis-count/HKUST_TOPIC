import gurobipy as grb
from gurobipy import GRB
import numpy as np
import copy

# This function uses deterministicModel.

class deterministicModel:
    def __init__(self, roll_width, given_lines, demand_width_array, I, s):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.s = s
        self.value_array = demand_width_array - self.s
        self.I = I

    def IP_formulation(self, demand_lower, demand_upper):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype = GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        if abs(sum(demand_upper)- 0)> 1e-4:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) <= demand_upper[i] for i in range(self.I))
        if abs(sum(demand_lower)- 0)> 1e-4:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) >= demand_lower[i] for i in range(self.I))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(
            self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        if m.status !=2:
            m.write('test.lp')
        # print('Optimal value of IP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newx = np.rint(newx)
        newd = np.sum(newx, axis=1)
        newd = np.rint(newd)
        return newd, newx

    # def IP_formulation2(self, roll_width, num_period, probab, seq):
    #     # used to test the booking performance when knowing the coming group
    #     demand = np.ceil(np.array(probab) * num_period)
    #     demand[seq-2] +=1
    #     m = grb.Model()

    #     x = m.addVars(self.I, len(roll_width), lb=0, vtype= GRB.INTEGER)
    #     m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
    #                               for i in range(self.I)) <= roll_width[j] for j in range(len(roll_width)))
    #     m.addConstrs(grb.quicksum(x[i, j] for j in range(
    #         len(roll_width))) <= demand[i] for i in range(self.I))

    #     m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(
    #         self.I) for j in range(len(roll_width))), GRB.MAXIMIZE)
    #     m.setParam('OutputFlag', 0)
    #     m.optimize()
    #     x_ij = np.array(m.getAttr('X'))
        
    #     newx = np.reshape(x_ij, (self.I, len(roll_width)))
    #     newd = np.sum(newx, axis=1)
    #     return newd, newx

    def LP_formulation(self, demand, roll_width):
        m = grb.Model()
        z = m.addVars(self.I, lb=0, vtype=GRB.CONTINUOUS)
        beta = m.addVars(self.given_lines, lb=0, vtype=GRB.CONTINUOUS)

        m.addConstrs(z[i] + beta[j] * (i+1 + self.s) >= i +
                     1 for j in range(self.given_lines) for i in range(self.I))

        m.setObjective(grb.quicksum(demand[i] * z[i] for i in range(
            self.I)) + grb.quicksum(roll_width[j] * beta[j] for j in range(
                self.given_lines)), GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        # m.write('bid_price.lp')
        x_ij = np.array(m.getAttr('X'))[-self.given_lines:]
        return x_ij, m.objVal

    # def LP_formulation2(self, demand, roll_width):
    #     #  used to test the performance of bid-price
    #     m = grb.Model()
    #     z = m.addVars(self.I, lb=0, vtype=GRB.CONTINUOUS)
    #     beta = m.addVars(self.given_lines, lb=0, vtype=GRB.CONTINUOUS)

    #     m.addConstrs(z[i] + beta[j] * (i+1 + self.s) >= i +
    #                  1 for j in range(self.given_lines) for i in range(self.I))

    #     m.setObjective(grb.quicksum(demand[i] * z[i] for i in range(
    #         self.I)) + grb.quicksum(roll_width[j] * beta[j] for j in range(
    #             self.given_lines)), GRB.MINIMIZE)
    #     m.setParam('OutputFlag', 0)
    #     m.optimize()
    #     x_ij = np.array(m.getAttr('X'))[-self.given_lines:]
    #     y_ij = np.array(m.getAttr('X'))[:self.I]
    #     return x_ij, y_ij, m.objVal

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

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()

        if m.status == GRB.OPTIMAL:
            return True
        else:
            return False

