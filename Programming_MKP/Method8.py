import gurobipy as grb
from gurobipy import GRB
import numpy as np
import copy

# This function uses column generation

class column_generation:
    def __init__(self, roll_width, given_lines, weight, I, value):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.weight = weight
        self.value_array = value
        self.I = I

    def dynamic_primal(self, dom_set, demand):
        # return optimal primal
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
                m.addConstr(x[i, j] <= grb.quicksum(dom_set[j][h][i] * y[j][h] for h in range(len(dom_set[j]))))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)

        m.setParam('OutputFlag', 0)
        m.optimize()
        # print(f'BPP :{m.objVal}')
        opt = np.array(m.getAttr('X'))
        opt_x = opt[:self.I * self.given_lines]
        opt_y = opt[self.I * self.given_lines:len(opt)]
        opt_x = np.reshape(opt_x, (self.I, self.given_lines))
        
        return opt_x, opt_y

    def dual_primal(self, dom_set, demand):
        # return optimal dual
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
                m.addConstr(x[i, j] <= grb.quicksum(dom_set[j][h][i] * y[j][h] for h in range(len(dom_set[j]))))

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

        m.addConstr(grb.quicksum(self.weight[i] * h[i] for i in range(self.I)) <= row_j)
        m.setObjective(grb.quicksum((self.value_array[i] - alpha[i]) * h[i] for i in range(self.I)) - gamma, GRB.MAXIMIZE)

        # m.write('1.lp')
        m.setParam('OutputFlag', 0)
        m.optimize()

        flag_new_column = m.objVal > 1e-10
        if flag_new_column:
            new_column = np.array(m.getAttr('X'))
        else:
            new_column = None

        return new_column

    def improved_bid(self, dom_set, demand):
        # return three dual items
        m = grb.Model()
        alpha = m.addVars(self.I, lb=0, vtype = GRB.CONTINUOUS, name = 'alpha')
        beta = m.addVars(self.I, self.given_lines, lb = 0, vtype = GRB.CONTINUOUS, name = 'beta')
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
        z = m.addVars(self.I, lb=0, vtype = GRB.CONTINUOUS)
        beta = m.addVars(self.given_lines, lb=0, vtype = GRB.CONTINUOUS)

        m.addConstrs(z[i] + beta[j] * self.weight[i] >= self.value_array[i] for j in range(self.given_lines) for i in range(self.I))

        m.setObjective(grb.quicksum(demand[i] * z[i] for i in range(self.I)) + grb.quicksum(
            roll_width[j] * beta[j] for j in range(self.given_lines)), GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        # m.write('bid_price.lp')
        alpha = np.array(m.getAttr('X'))[:self.I]
        beta_j = np.array(m.getAttr('X'))[-self.given_lines:]
        # print(f'alpha: {alpha}')
        # print(beta_j)
        return m.objVal, beta_j

    def setGeneration(self, dom_set, demand, roll_width):
        # New_pattern: List[]
        # dom_set: Initial Set
        flag_new_pattern = True
        count = 0
        while flag_new_pattern:
            count += 1
            if count > 50:
                break

            #  return the dual of the master problem
            dual_alpha, dual_gamma = self.dual_primal(dom_set, demand)
            # print(f'alpha: {dual_alpha}')
            # print(f'gamma: {dual_gamma}')

            add_count = 0
            for j in range(self.given_lines):
                new_pattern = self.subproblem(dual_alpha, dual_gamma[j], roll_width[j])
                # print(new_pattern)
                if new_pattern is not None:
                    dom_set[j] = np.vstack((dom_set[j], new_pattern))
                    add_count += 1
            if  add_count == 0:
                flag_new_pattern = False
        opt_x, opt_y = self.dynamic_primal(dom_set, demand)
        # for j in range(self.given_lines):
        #     print(f'set{j} have: {dom_set[j]}')
        return opt_x, opt_y

    def setGeneration_bid(self, dom_set, demand, roll_width):
        # New_pattern: List[]
        # dom_set: Initial Set
        flag_new_pattern = True
        count = 0
        while flag_new_pattern:
            count += 1
            if count > 50:
                break

            #  return the dual of the master problem
            dual_alpha, dual_gamma = self.dual_primal(dom_set, demand)
            # print(f'alpha: {dual_alpha}')
            # print(f'gamma: {dual_gamma}')

            add_count = 0
            for j in range(self.given_lines):
                new_pattern = self.subproblem(dual_alpha, dual_gamma[j], roll_width[j])
                # print(new_pattern)
                if new_pattern is not None:
                    dom_set[j] = np.vstack((dom_set[j], new_pattern))
                    add_count += 1
            if add_count == 0:
                flag_new_pattern = False
        alpha, beta, gamma = self.improved_bid(dom_set, demand)
        # for j in range(self.given_lines):
        #     print(f'set{j} have: {dom_set[j]}')
        return alpha, beta, gamma, dom_set


if __name__ == "__main__":
    given_lines = 4
    # roll_width = [21, 22, 23, 24, 25, 26, 27, 28]
    roll_width = [7, 8, 8, 4]

    I = 3  # the number of group types
    value = np.array([4, 6, 8])
    # value = np.array([2])

    # value = np.array([2, 4, 7])
    weight = np.array([3, 4, 5])

    # weight = np.array([2])
    # value = weight

    demand_array = np.array([2, 4, 2])

    test = column_generation(roll_width, given_lines, weight, I, value)

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
    #     dom_set[j][0][-1] = roll_width[j] // weight[-1]  # 直接修改

    # dual1, dual2 = test.dual_primal(dom_set, demand)
    # print(dual1)
    # print(np.array(list(dual1.values())))
    # pattern = test.subproblem(dual1, dual2[0], 5)

    ############# Primal ###################
    opt_x, opt_y = test.setGeneration(dom_set, demand_array, roll_width)
    # print(f'x: {opt_x}')
    # print(f'y: {opt_y}')
    ############# END ###################

    ############ BPC #################
    obj, beta_j = test.LP_formulation(demand_array, roll_width)
    # print(obj)
    ############ END #################

    ############ BPP #################
    alpha, beta, domset = test.setGeneration_bid(dom_set, demand_array, roll_width)
    print(f'alpha: {alpha}')
    print(f'beta: {beta}')
    print(f'set: {domset}')
    ############ END #################

    # for i in range(I):
    #     print(beta[i] - weight[i] * beta_j[i])
