from msilib.schema import Class
from xmlrpc.client import gzip_decode
import gurobipy as grb
from gurobipy import GRB
import numpy as np

#  This class is used to obtain the deterministic model

class deterministicModel:
    def __init__(self, roll_width, given_lines, demand_width_array, demand_number_array) -> None:
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_number_array = demand_number_array
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array - 1
        self.column = np.diag(np.floor(self.roll_width / self.demand_width_array))

    def master_problem(self, vtype):
        m = grb.Model()
        x = m.addMVar(shape = self.column.shape[1], lb=0, vtype=vtype)
        m.addConstr(lhs = x.sum() <= self.given_lines)
        m.addConstr(lhs = self.column @ x <= self.demand_number_array)
        h_k = self.value_array @ self.column
        m.setObjective(h_k @ x, GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()

        if vtype == GRB.CONTINUOUS:
            return np.array(m.getAttr('Pi', m.getConstrs()))
        else:
            return m.objVal, np.array(m.getAttr('X'))


    def lp_relaxation(self):
        m = grb.Model()
        x = m.addMVar(shape = self.column.shape[1], lb=0, vtype= GRB.CONTINUOUS)
        m.addConstr(lhs = x.sum() <= self.given_lines)
        m.addConstr(lhs = self.column @ x <= self.demand_number_array)
        h_k = self.value_array @ self.column
        m.setObjective(h_k @ x, GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        return m.objVal, np.array(m.getAttr('X'))

    def restricted_lp_master_problem(self):
        return self.master_problem(GRB.CONTINUOUS)

    def restricted_ip_master_problem(self):
        return self.master_problem(GRB.INTEGER)

    def knapsack_subproblem(self, kk):
        m = grb.Model()
        x = m.addMVar(shape= kk.shape[0]-1, lb = 0, vtype = GRB.INTEGER)
        m.addConstr(lhs= self.demand_width_array @ x <= self.roll_width)
        new_kk = np.delete(kk,0)
        m.setObjective((self.value_array - new_kk) @ x-kk[0], GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()

        flag_new_column = m.objVal > 1e-10
        if flag_new_column:
            new_column = m.getAttr('X')
        else:
            new_column = None
        return flag_new_column, new_column


    def columnGeneration(self):
        flag_new_cut_pattern = True
        new_cut_pattern = None
        cut_pattern = self.column
        count = 0
        while flag_new_cut_pattern:
            count += 1
            if count > 20:
                break
            if new_cut_pattern:
                cut_pattern = np.column_stack((cut_pattern, new_cut_pattern))
            self.column = cut_pattern
            kk = self.restricted_lp_master_problem()
            flag_new_cut_pattern, new_cut_pattern = self.knapsack_subproblem(kk)
        # maximum_value, optimal_number = restricted_ip_master_problem(cut_pattern)
        lp_result,lp_solution = self.lp_relaxation()

        print('************************************************')
        print('parameter:')
        print(f'roll_width: {self.roll_width}')
        print(f'demand_width_array: {self.demand_width_array}')
        print(f'demand_number_array: {self.demand_number_array}')
        print('result:')
        # print(f'maximum_value: {maximum_value}')
        print(f'cut_pattern: \n{cut_pattern}')
        # print(f'optimal_number: \n {optimal_number}')
        print('************************************************')
        print(f'lp_result: \n {lp_result}')
        print(f'lp_solution: \n {lp_solution}')

        print(f'Supply: \n {cut_pattern @ lp_solution}')
        # return maximum_value,lp_result

    def IP_formulation(self):
        n = len(self.demand_width_array)
        m = grb.Model()
        x = m.addVars(n, self.given_lines, lb= 0, vtype= GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i,j] for i in range(n)) <= self.roll_width for j in range(self.given_lines))
        m.addConstrs(grb.quicksum(x[i,j] for j in range(self.given_lines)) <= self.demand_number_array[i] for i in range(n))
        m.setObjective(grb.quicksum(self.value_array[i] * x[i,j] for i in range(n) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        print('************************************************')
        print('Optimal value of IP is: %g' % m.objVal)
        x_ij = m.getAttr('X')
        cut_p = np.array(x_ij).reshape((n, self.given_lines))
        print(cut_p)
        print(cut_p.sum(axis=1))
        return m.objVal

    def LP_formulation(self):
        n = len(self.demand_width_array)
        m = grb.Model()
        x = m.addVars(n, self.given_lines, lb= 0, vtype= GRB.CONTINUOUS)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i,j] for i in range(n)) <= self.roll_width for j in range(self.given_lines))
        m.addConstrs(grb.quicksum(x[i,j] for j in range(self.given_lines)) <= self.demand_number_array[i] for i in range(n))
        m.setObjective(grb.quicksum(self.value_array[i] * x[i,j] for i in range(n) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        print('************************************************')
        print('Optimal value of LP is: %g' % m.objVal)
        x_ij = m.getAttr('X')
        print(np.array(x_ij).reshape((n, self.given_lines)))
        return m.objVal

if  __name__ == '__main__':
    # In our case, the roll width needs to increase by 1.
    roll_width = np.array(18) # (N+1) seats
    demand_width_array = np.array([3, 4, 6, 9, 10]) # [1,3,5,6] people in each group
    # value_array = demand_width_array - 1
    # initial_cut_pattern = np.diag(np.floor(roll_width / demand_width_array))
    demand_number_array = np.array([2, 1, 2, 1, 1])
    # n = len(demand_width_array)
    given_lines = 2

    my = deterministicModel(roll_width, given_lines, demand_width_array, demand_number_array)
    my.columnGeneration()
    my.LP_formulation()
