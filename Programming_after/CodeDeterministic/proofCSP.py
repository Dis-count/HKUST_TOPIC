import gurobipy as grb
from gurobipy import GRB
import numpy as np
import math

# This funtion finds the gap between cutting stock and its integer results.

def master_problem(column, vtype):
    m = grb.Model()
    x = m.addMVar(shape = column.shape[1], lb=0, vtype=vtype)
    m.addConstr(lhs = column @ x >= demand_number_array)
    m.setObjective(x.sum(), GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()

    if vtype == GRB.CONTINUOUS:
        return np.array(m.getAttr('Pi', m.getConstrs()))
    else:
        return m.objVal, np.array(m.getAttr('X'))

def lp_relaxation(column):
    m = grb.Model()
    x = m.addMVar(shape = column.shape[1], lb=0, vtype= GRB.CONTINUOUS)
    m.addConstr(lhs = column @ x >= demand_number_array)
    m.setObjective(x.sum(), GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()
    return m.objVal, np.array(m.getAttr('X'))

def restricted_lp_master_problem(column):
    return master_problem(column, GRB.CONTINUOUS)

def restricted_ip_master_problem(column):
    return master_problem(column, GRB.INTEGER)

def knapsack_subproblem(kk):
    m = grb.Model()
    x = m.addMVar(shape=kk.shape[0], lb=0, vtype=GRB.INTEGER)
    m.addConstr(lhs= demand_width_array @ x <= roll_width)
    m.setObjective(1 - kk @ x, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()

    flag_new_column = m.objVal < -1e-10
    if flag_new_column:
        new_column = m.getAttr('X')
    else:
        new_column = None
    return flag_new_column, new_column

def columnGeneration():
    flag_new_cut_pattern = True
    new_cut_pattern = None
    cut_pattern = initial_cut_pattern
    while flag_new_cut_pattern:
        if new_cut_pattern:
            cut_pattern = np.column_stack((cut_pattern, new_cut_pattern))
        kk = restricted_lp_master_problem(cut_pattern)
        flag_new_cut_pattern, new_cut_pattern = knapsack_subproblem(kk)
    minimal_stock, optimal_number = restricted_ip_master_problem(cut_pattern)
    lp_result,lp_solution = lp_relaxation(cut_pattern)
    # print('************************************************')
    # print('parameter:')
    # print(f'roll_width: {roll_width}')
    # print(f'demand_width_array: {demand_width_array}')
    # print(f'demand_number_array: {demand_number_array}')
    # print('result:')
    # print(f'minimal_stock: {minimal_stock}')
    # print(f'cut_pattern: \n{cut_pattern}')
    # print(f'optimal_number: \n {optimal_number}')
    # print('************************************************')
    # print(f'lp_result: \n {lp_result}')
    # print(f'lp_solution: \n {lp_solution}')
    # print(f'Supply: \n {cut_pattern @ lp_solution}')
    return lp_result

def IP_formulation(n, k_upper):
    m = grb.Model()
    x = m.addVars(n, k_upper, lb= 0, vtype= GRB.INTEGER)
    y = m.addVars(k_upper, lb= 0, vtype= GRB.BINARY)
    m.addConstrs(grb.quicksum(demand_width_array[i] * x[i,j] for i in range(n)) <= roll_width* y[j] for j in range(k_upper))
    m.addConstrs(grb.quicksum(x[i,j] for j in range(k_upper)) >= demand_number_array[i] for i in range(n))
    m.setObjective(grb.quicksum(y[j] for j in range(k_upper)), GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()
    # print('************************************************')
    # print('Optimal value of IP is: %g' % m.objVal)
    return m.objVal


for k in range(1000):
    # In our case, the roll width needs to increase by 1.
    # roll_width = np.random.randint(15,30)
    roll_width  = np.array(22)
    demand_width_array = np.array([3, 4, 5])

    initial_cut_pattern = np.diag(np.floor(roll_width / demand_width_array))
    demand_number_array = np.random.choice(12, 3)
    # demand_number_array = np.array([2, 2, 4])
    # print('************************************************')
    # print('parameter:')
    # print(f'roll_width: {roll_width}')
    # print(f'demand_width_array: {demand_width_array}')
    # print(f'demand_number_array: {demand_number_array}')
    n = len(demand_width_array)
    lp_result = columnGeneration()
    k_upper = int(lp_result+2)

    b = IP_formulation(n, k_upper)

    if b - math.ceil(lp_result)> 0.5:
        print(b)
        print(lp_result)
        print('*******')
