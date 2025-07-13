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

    def dynamic_primal(self, dom_set, demand):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype=GRB.CONTINUOUS, name = 'x')
        y1 = m.addVars(self.given_lines, len(
            dom_set[0]), vtype=GRB.CONTINUOUS, name='y1')
        y2 = m.addVars(self.given_lines, len(dom_set[1]),  vtype=GRB.CONTINUOUS)

        m.addConstr(grb.quicksum(y1[0, h] for h in range(len(dom_set[0]))) <= 1)

        m.addConstr(grb.quicksum(y2[1, h] for h in range(len(dom_set[1]))) <= 1)

        # m.addConstr(grb.quicksum(y[j, h] for h in range(len(dom_set))) <= 1 for j in range(self.given_lines))

        m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) <= demand[i] for i in range(self.I))

        for i in range(self.I):
            term = 0
            for h_index, h in enumerate(dom_set[0]):
                term += h[i] * y1[0, h_index]
            m.addConstr(x[i, 0] == term)

        for i in range(self.I):
            term = 0
            for h_index, h in enumerate(dom_set[1]):
                term += h[i] * y2[1, h_index]
            m.addConstr(x[i, 1] == term)

        # m.addConstrs(x[i, j] == grb.quicksum(dom_set[h][i] * y[j, h] for h in range(len(dom_set)))
        #              for i in range(self.I) for j in range(self.given_lines))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)

        m.setParam('OutputFlag', 0)
        m.optimize()
        
        m.write('2.lp')
        dual = np.array(m.getAttr('X'))
        opt_x = dual[0: self.I * self.given_lines]
        opt_x = np.reshape(opt_x, (self.I, self.given_lines))
        
        return opt_x

    def subproblem(self, alpha, gamma, row_j):
        m = grb.Model()
        h = m.addVars(self.I, lb=0, vtype = GRB.INTEGER, name='h')

        m.addConstr(grb.quicksum(h[i] for i in range(self.I)) <= row_j)

        m.setObjective(grb.quicksum((self.value_array[i] - alpha[i]) * h[i] for i in range(
            self.I)) - gamma, GRB.MAXIMIZE)

        m.setParam('OutputFlag', 0)
        m.optimize()

        m.write('3.lp')
        dual = np.array(m.getAttr('X'))

        return dual


    def improved_bid(self, dom_set, demand):
        m = grb.Model()
        alpha = m.addVars(self.I, lb=0, vtype=GRB.CONTINUOUS, name = 'alpha')
        beta = m.addVars(self.I, self.given_lines,
                         lb = float('-inf'), vtype=GRB.CONTINUOUS, name = 'beta')
        gamma = m.addVars(self.given_lines, lb=0, vtype=GRB.CONTINUOUS, name = 'gamma')

        m.addConstrs(alpha[i] + beta[i, j] >= self.demand_width_array[i] - self.s
                     for i in range(self.I) for j in range(self.given_lines))

        # m.addConstrs(grb.quicksum(beta[i, j] * dom_set[h][i] for i in range(self.I)) <= gamma[j]
        #              for j in range(self.given_lines) for h in range(len(dom_set[j])) )

        for j in range(self.given_lines):
            for coff_h in dom_set[j]:
                m.addConstr(grb.quicksum(beta[i, j] * coff_h[i] for i in range(self.I)) <= gamma[j])

        m.setObjective(grb.quicksum(demand[i] * alpha[i] for i in range(self.I)) + grb.quicksum(gamma[j] for j in range(self.given_lines)), GRB.MINIMIZE)

        m.setParam('OutputFlag', 0)
        m.write('1.lp')
        m.optimize()

        dual = np.array(m.getAttr('X'))
        opt_alpha = dual[:self.I]
        opt_beta = dual[self.I: self.I * self.given_lines + self.I]
        opt_beta = np.reshape(opt_beta, (self.I, self.given_lines))
        opt_gamma = dual[self.I * self.given_lines + self.I:]

        return opt_alpha, opt_beta, opt_gamma


    def LP_formulation(self, demand, roll_width):
        #  The traditional bid-price control
        m = grb.Model()
        z = m.addVars(self.I, lb=0, vtype=GRB.CONTINUOUS)
        beta = m.addVars(self.given_lines, lb=0, vtype=GRB.CONTINUOUS)

        m.addConstrs(z[i] + beta[j] * (i+1 + self.s) >= i +
                     1 for j in range(self.given_lines) for i in range(self.I))

        m.setObjective(grb.quicksum(demand[i] * z[i] for i in range(self.I)) + grb.quicksum(
            roll_width[j] * beta[j] for j in range(self.given_lines)), GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        # m.write('bid_price.lp')
        alpha = np.array(m.getAttr('X'))[:self.I]
        x_ij = np.array(m.getAttr('X'))[-self.given_lines:]
        print(f'alpha: {alpha}')
        print(x_ij)
        return x_ij, m.objVal

    # def 

if __name__ == "__main__":
    given_lines = 2
    # roll_width = np.ones(given_lines) * 5
    roll_width = [5, 6]
    I = 4  # the number of group types
    s = 1
    demand_width_array = np.arange(2, 2+I)

    demand = np.array([4, 1, 2, 0.8])

    test = deterministicModel(roll_width, given_lines, demand_width_array, I, s)

    # dom_set = [[0,0,0,4],
    #            [0,0,4,1],
    #            [0,1,2,2],
    #            [0,3,3,0],
    #            [1,0,1,3],
    #            [0,2,0,3],
    #            [2,1,1,2],
    #            [1,2,2,1]]
    dom_set = [[[0, 0, 0, 1],
               [0, 0, 1, 0],
               [1, 1, 0, 0],
               [2, 0, 0, 0]],
               [[0, 0, 0, 1],
               [1, 0, 1, 0],
               [0, 2, 0, 0],
               [3, 0, 0, 0],
               [1, 1, 0, 0]]]
    # print(dom_set[0][0][0])

    # opt_alpha, opt_beta, opt_gamma = test.improved_bid(dom_set, demand)

    opt_x = test.dynamic_primal(dom_set, demand)

    # print(f'alpha: {opt_alpha}')
    # print(f'beta: {opt_beta}')
    # print(f'gamma: {opt_gamma}')

    print(f'x: {opt_x}')

    test.LP_formulation(demand, roll_width)
