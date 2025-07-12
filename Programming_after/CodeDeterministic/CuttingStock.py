import gurobipy as grb
from gurobipy import GRB
import numpy as np
import copy

# the basic cutting stock by column generation

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


# In our case, the roll width needs to increase by 1.
roll_width = np.array(18) # (N+1) seats
demand_width_array = np.array([4, 6, 9, 10]) # [1,3,5,6] people in each group
demand_number_array = np.array([1, 2, 1, 1])
demand_number = copy.deepcopy(demand_number_array) # keep unchanged

initial_cut_pattern = np.diag(np.floor(roll_width / demand_width_array))
m = len(demand_width_array)
n = 10
# roll_width = np.array(31) # 20 seats
# demand_width_array = np.array([2, 3, 4, 5, 6, 7, 8, 9,10]) # [1,2,3,5,6] people in each group
# demand_number_array = np.array([10, 15, 17, 20, 20, 15,10,8,7])
# initial_cut_pattern = np.diag(np.floor(roll_width / demand_width_array))

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

    lp_result, lp_solution = lp_relaxation(cut_pattern)
    print('************************************************')
    print('parameter:')
    print(f'roll_width: {roll_width}')
    print(f'demand_width_array: {demand_width_array}')
    print(f'demand_number_array: {demand_number_array}')
    print('result:')
    print(f'minimal_stock: {minimal_stock}')
    print(f'cut_pattern: \n{cut_pattern}')
    print(f'optimal_number: \n {optimal_number}')
    print(f'lp_result: \n {lp_result}')
    print(f'lp_solution: \n {lp_solution}')

    # return cut_pattern,optimal_number
columnGeneration()
# total_demand = eff_pattern @ eff_number

# Sort the dominance solution and give the new demand
def sortSolution(n):
    # Find the efficient number
    a = optimal_number !=0
    a.tolist()
    eff_pattern = cut_pattern[:,a]
    eff_number = optimal_number[a]

    capacityTaken = demand_width_array @ eff_pattern
    valueTaken = capacityTaken - eff_pattern.sum(axis=0) +1
    orderMatrix = np.row_stack((eff_pattern,eff_number,valueTaken)) # Add two extra rows.
    afterSort = orderMatrix[:,np.argsort(-orderMatrix[m+1])]
    afterPattern = afterSort[0:m]
    cumuNumber = np.cumsum(afterSort[m])
    k = 0
    for i in cumuNumber:
        if n <= i:
            break
        k += 1
    finalPattern = afterPattern[:,0:k+1]
    finalNumber = np.append(afterSort[m][0:k], n- cumuNumber[k-1])
    finalResult = np.row_stack((finalPattern, finalNumber))
    print(f'Result: \n{finalResult}')
    satisfiedDemand = finalPattern @ finalNumber
    minDemand = np.minimum(satisfiedDemand, demand_number)
    totalValue = minDemand @ (demand_width_array-1)
    # Increase the largest demand to total demand.
    for i in range(m):
        if minDemand[::-1][i] < demand_number[::-1][i]:
           minDemand[-1-i] = demand_number[-1-i]
           break
    return minDemand,totalValue,finalResult


# gap = demand_number_array - finalPattern @ finalNumber

minDemand,totalValue,finalResult = sortSolution(n)

optimalValue = 0

while totalValue > optimalValue:
    optimalValue = totalValue
    result = finalResult
    # Update the demand
    demand_number_array = minDemand

    cut_pattern,optimal_number = columnGeneration()
    minDemand,totalValue,finalResult = sortSolution(n)

final_Demand = result[0:m] @ result[-1]
print(f'Satisfied_Demand: \n{final_Demand}')
print(f'Final_Result: \n{result}')
print(f'Total_Value: {optimalValue}')
