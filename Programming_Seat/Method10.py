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
        #  Seat planning with upper and lower bound
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb = 0, vtype = GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        if abs(sum(demand_upper)- 0)> 1e-4:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) <= demand_upper[i] for i in range(self.I))
        if abs(sum(demand_lower)- 0)> 1e-4:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) >= demand_lower[i] for i in range(self.I))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(
            self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        # m.setParam('TimeLimit', 40)
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

    def LP(self, demand_upper):
        #  Seat planning with upper and lower bound
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb = 0, vtype = GRB.CONTINUOUS)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))

        m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) <= demand_upper[i] for i in range(self.I))


        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        # m.setParam('TimeLimit', 40)
        m.optimize()
        if m.status != 2:
            m.write('test.lp')
        # print('Optimal value of IP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        # newx = np.rint(newx)
        newd = np.sum(newx, axis=1)
        # newd = np.rint(newd)
        return newd, newx


    def IP_advanced(self, demand_lower):
        #  seat planning given the lower demand
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype=GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))

        for k in range(self.I):
            m.addConstr(grb.quicksum(x[i, j] for j in range(self.given_lines) for i in range(self.I-k-1, self.I, 1)) >= grb.quicksum(demand_lower[i] for i in range(self.I-k-1, self.I, 1)))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()

        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)

        return newd, newx

    def LP_formulation(self, demand, roll_width):
        #  Dual of the bid-price 
        m = grb.Model()
        z = m.addVars(self.I, lb = 0, vtype = GRB.CONTINUOUS)
        beta = m.addVars(self.given_lines, lb = 0, vtype = GRB.CONTINUOUS)

        m.addConstrs(z[i] + beta[j] * (i+1 + self.s) >= i + 1 for j in range(self.given_lines) for i in range(self.I))

        m.setObjective(grb.quicksum(demand[i] * z[i] for i in range(self.I)) + grb.quicksum(roll_width[j] * beta[j] for j in range(self.given_lines)), GRB.MINIMIZE)
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


if __name__ == "__main__":
    given_lines = 6
    roll_width = np.ones(given_lines) * 21
    I = 4  # the number of group types
    s = 1
    demand_width_array = np.arange(2, 2+I)

    demand = np.array([18, 19, 20, 15])

    test = deterministicModel(roll_width, given_lines,demand_width_array, I, s)

    d0, newx = test.IP_formulation(np.zeros(I), demand)

    d1, x = test.LP(demand)

    print(f'IP solution: {newx} supply: {d0}')
    print(f'LP solution: {x} supply: {d1}')
