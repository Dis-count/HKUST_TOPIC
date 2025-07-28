import gurobipy as grb
from gurobipy import GRB
import numpy as np
import copy

# This function uses column generation

class column_generation:
    def __init__(self, roll_width, given_lines, demand_width_array, I, s):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.s = s
        self.value_array = demand_width_array - self.s
        self.I = I

    def dynamic_primal(self, dom_set, demand):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb = 0, vtype = GRB.CONTINUOUS, name = 'x')

        y = {}  # 存储所有 y_i

        for i in range(self.given_lines):  # i 是 y_i 的索引
            # y_i 是一个二维变量: 行数=self.given_lines, 列数=len(dom_set[i])
            y[i] = m.addVars(len(dom_set[i]), vtype = GRB.CONTINUOUS, name = f'y_{i}')
        
        # 访问方式：
        # y[0][0] 表示 y0 第1列变量
        # y[1][2] 表示 y1 第3列变量

        m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) <= demand[i] for i in range(self.I))

        m.addConstrs(grb.quicksum(y[j][h] for h in range(len(dom_set[j]))) <= 1 for j in range(self.given_lines))

        for i in range(self.I):
            for j in range(self.given_lines):
                m.addConstr(x[i, j] == grb.quicksum(dom_set[j][h][i] * y[j][h] for h in range(len(dom_set[j]))))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)

        m.setParam('OutputFlag', 0)
        m.optimize()
        
        opt = np.array(m.getAttr('X'))
        opt_x = opt[:self.I * self.given_lines]
        opt_y = opt[self.I * self.given_lines:len(opt)]
        opt_x = np.reshape(opt_x, (self.I, self.given_lines))
        
        return opt_x, opt_y

    def dual_primal(self, dom_set, demand):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb = 0, vtype = GRB.CONTINUOUS, name='x')

        y = {}  # 存储所有 y_i

        for i in range(self.given_lines):  # i 是 y_i 的索引
            # y_i 是一个二维变量: 行数=self.given_lines, 列数=len(dom_set[i])
            y[i] = m.addVars(len(dom_set[i]), vtype=GRB.CONTINUOUS, name= f'y_{i}')

        # 访问方式：
        # y[0][0] 表示 y0 第1列变量
        # y[1][2] 表示 y1 第3列变量

        alpha_con = m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) <= demand[i] for i in range(self.I))

        gamma_con = m.addConstrs(grb.quicksum(y[j][h] for h in range(len(dom_set[j]))) <= 1 for j in range(self.given_lines))

        for i in range(self.I):
            for j in range(self.given_lines):
                m.addConstr(x[i, j] == grb.quicksum(dom_set[j][h][i] * y[j][h] for h in range(len(dom_set[j]))))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)

        m.setParam('OutputFlag', 0)
        m.optimize()

        # m.write('2.lp')

        dual1 = m.getAttr('Pi', alpha_con)
        dual2 = m.getAttr('Pi', gamma_con)

        dual_alpha = np.array(list(dual1.values()))
        dual_gamma = np.array(list(dual2.values()))

        return dual_alpha, dual_gamma

    def subproblem(self, alpha, gamma, row_j):
        m = grb.Model()
        h = m.addVars(self.I, lb = 0, vtype = GRB.INTEGER, name='h')

        m.addConstr(grb.quicksum(self.demand_width_array[i] * h[i] for i in range(self.I)) <= row_j)
        m.setObjective(grb.quicksum((self.value_array[i] - alpha[i]) * h[i] for i in range(self.I)) - gamma, GRB.MAXIMIZE)

        # m.write('11.lp')
        m.setParam('OutputFlag', 0)
        m.optimize()

        flag_new_column = m.objVal > 1e-10
        if flag_new_column:
            new_column = np.array(m.getAttr('X'))
        else:
            new_column = None

        return new_column

    def improved_bid(self, dom_set, demand):
        m = grb.Model()
        alpha = m.addVars(self.I, lb=0, vtype = GRB.CONTINUOUS, name = 'alpha')
        beta = m.addVars(self.I, self.given_lines, lb = float('-inf'), vtype = GRB.CONTINUOUS, name = 'beta')
        gamma = m.addVars(self.given_lines, lb=0, vtype = GRB.CONTINUOUS, name = 'gamma')

        m.addConstrs(alpha[i] + beta[i, j] >= self.value_array[i] for i in range(self.I) for j in range(self.given_lines))

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

    def setGeneration(self, dom_set, demand, roll_width):
        # New_pattern: List[]
        # dom_set: Initial Set
        flag_new_pattern = True
        count = 0
        while flag_new_pattern:
            count += 1
            if count > 20:
                break

            #  return the dual of the master problem
            dual_alpha, dual_gamma = self.dual_primal(dom_set, demand)
            print(f'alpha: {dual_alpha}')
            print(f'gamma: {dual_gamma}')

            add_count = 0
            for j in range(self.given_lines):
                new_pattern = self.subproblem(dual_alpha, dual_gamma[j], roll_width[j])
                print(new_pattern)
                if new_pattern is not None:
                    dom_set[j] = np.vstack((dom_set[j], new_pattern))
                    add_count += 1
            if  add_count == 0:
                flag_new_pattern = False
        opt_x, opt_y = self.dynamic_primal(dom_set, demand)
        # for j in range(self.given_lines):
        #     print(f'set{j} have: {len(dom_set[j])}')
        return opt_x, opt_y


if __name__ == "__main__":
    given_lines = 8
    # roll_width = np.ones(given_lines) * 21
    roll_width = [25, 26, 21, 22, 23, 24, 27, 28]
    I = 4  # the number of group types
    s = 1
    demand_width_array = np.arange(2, 2+I)

    demand = np.array([4, 12, 12, 21])

    test = column_generation(roll_width, given_lines, demand_width_array, I, s)

    # dom_set = [[[0, 0, 0, 1],
    #            [0, 0, 1, 0],
    #            [1, 1, 0, 0],
    #            [2, 0, 0, 0]],
    #            [[0, 0, 0, 1],
    #            [1, 0, 1, 0],
    #            [0, 2, 0, 0],
    #            [3, 0, 0, 0],
    #            [1, 1, 0, 0]]]

    dom_set = [np.zeros((1, I)) for _ in range(given_lines)]  # 初始化为 (1, I) 的全零数组

    # for j in range(given_lines):
    #     dom_set[j][0][-1] = roll_width[j] // demand_width_array[-1]  # 直接修改

    # dual1, dual2 = test.dual_primal(dom_set, demand)
    # print(dual1)
    # print(np.array(list(dual1.values())))

    # pattern = test.subproblem(dual1, dual2[0], 5)

    opt_x, opt_y = test.setGeneration(dom_set, demand, roll_width)

    print(f'x: {opt_x}')
    print(f'y: {opt_y}')
