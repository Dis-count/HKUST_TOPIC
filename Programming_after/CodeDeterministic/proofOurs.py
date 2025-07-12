import gurobipy as grb
from gurobipy import GRB
import numpy as np

# This funtion finds the gap between LP- seating and its integer results.

def master_problem(column, vtype):
    m = grb.Model()
    x = m.addMVar(shape = column.shape[1], lb=0, vtype=vtype)
    m.addConstr(lhs = x.sum() <= given_lines)
    m.addConstr(lhs = column @ x <= demand_number_array)
    h_k = value_array @ column
    m.setObjective(h_k @ x, GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()

    if vtype == GRB.CONTINUOUS:
        return np.array(m.getAttr('Pi', m.getConstrs()))
    else:
        return m.objVal, np.array(m.getAttr('X'))

def lp_relaxation(column):
    m = grb.Model()
    x = m.addMVar(shape = column.shape[1], lb=0, vtype= GRB.CONTINUOUS)
    m.addConstr(lhs = x.sum() <= given_lines)
    m.addConstr(lhs = column @ x <= demand_number_array)
    h_k = value_array @ column
    m.setObjective(h_k @ x, GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()
    return m.objVal, np.array(m.getAttr('X'))

def restricted_lp_master_problem(column):
    return master_problem(column, GRB.CONTINUOUS)

def restricted_ip_master_problem(column):
    return master_problem(column, GRB.INTEGER)

def knapsack_subproblem(kk):
    m = grb.Model()
    x = m.addMVar(shape= kk.shape[0]-1, lb=0, vtype=GRB.INTEGER)
    m.addConstr(lhs= demand_width_array @ x <= roll_width)
    new_kk = np.delete(kk,0)
    m.setObjective((value_array - new_kk) @ x-kk[0], GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()

    flag_new_column = m.objVal > 1e-10
    if flag_new_column:
        new_column = m.getAttr('X')
    else:
        new_column = None
    return flag_new_column, new_column

def columnGeneration():
    flag_new_cut_pattern = True
    new_cut_pattern = None
    cut_pattern = initial_cut_pattern
    count = 0
    while flag_new_cut_pattern:
        count += 1
        if count > 20:
            break
        if new_cut_pattern:
            cut_pattern = np.column_stack((cut_pattern, new_cut_pattern))
        kk = restricted_lp_master_problem(cut_pattern)
        flag_new_cut_pattern, new_cut_pattern = knapsack_subproblem(kk)
    maximum_value, optimal_number = restricted_ip_master_problem(cut_pattern)
    lp_result,lp_solution = lp_relaxation(cut_pattern)

    # print('result:')
    # # print(f'maximum_value: {maximum_value}')
    # print(f'cut_pattern: \n{cut_pattern}')
    # # print(f'optimal_number: \n {optimal_number}')
    # print('************************************************')
    # print(f'lp_result: \n {lp_result}')
    # print(f'lp_solution: \n {lp_solution}')
    return lp_result

def IP_formulation(n,given_lines):
    m = grb.Model()
    x = m.addVars(n,given_lines, lb= 0, vtype= GRB.INTEGER)
    m.addConstrs(grb.quicksum(demand_width_array[i] * x[i,j] for i in range(n)) <= roll_width for j in range(given_lines))
    m.addConstrs(grb.quicksum(x[i,j] for j in range(given_lines)) <= demand_number_array[i] for i in range(n))
    m.setObjective(grb.quicksum(value_array[i] * x[i,j] for i in range(n) for j in range(given_lines)), GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()
    # print('Optimal value of IP is: %g' % m.objVal)
    x_ij = m.getAttr('X')
    # print(np.array(x_ij).reshape((n,given_lines)))
    return m.objVal

for k in range(1000):
    # In our case, the roll width needs to increase by 1.
    roll_width = np.random.randint(16,41)
    demand_width_array = np.array([2, 3, 4, 5, 6, 7])
    # demand_width_array = np.random.choice(8, 4, replace=False)+1 # [1,3,5,6] people in each group
    # value_array = np.random.choice(10, 4, replace=False) # [1,3,5,6] people in each group
    value_array = demand_width_array - 1
    initial_cut_pattern = np.diag(np.floor(roll_width / demand_width_array))
    demand_number_array = np.random.choice(10, 6, replace=False)+10
    # print('************************************************')
    # print('parameter:')
    # print(f'roll_width: {roll_width}')
    # print(f'demand_width_array: {demand_width_array}')
    # print(f'demand_number_array: {demand_number_array}')
    n = len(demand_width_array)
    given_lines = 10
    a = IP_formulation(n,given_lines)
    b = columnGeneration()
    if int(b) - a> 0.5:
        print(int(b) - a)
