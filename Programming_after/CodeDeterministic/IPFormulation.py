import gurobipy as grb
from gurobipy import GRB
import numpy as np
import time

def IP_formulation(n, given_lines):
    m = grb.Model()
    x = m.addVars(n, given_lines, lb= 0, vtype= GRB.INTEGER)
    m.addConstrs(grb.quicksum(demand_width_array[i] * x[i,j] for i in range(n)) <= roll_width[j] for j in range(given_lines))
    m.addConstrs(grb.quicksum(x[i,j] for j in range(given_lines)) <= demand_number_array[i] for i in range(n))
    m.setObjective(grb.quicksum(value_array[i] * x[i,j] for i in range(n) for j in range(given_lines)), GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.write('tt.lp')
    m.optimize()
    print('************************************************')
    print('Optimal value of IP is: %g' % m.objVal)
    x_ij = m.getAttr('X')
    sol = np.array(x_ij).reshape((n, given_lines))
    print(np.array(x_ij).reshape((n, given_lines)))
    return m.objVal,sol


def generate_assignment(solution):
    group_size = [i+1 for i in range(len(solution))]
    column = np.array(solution)[:, 0]
    seat_assignment = []
    for i, k in enumerate(column):
        for j in range(k):
            seat_assignment.append(group_size[i])
    return seat_assignment

if __name__ == '__main__':
    given_lines = 150
    roll_width = np.random.randint(low = 151, high = 201, size = given_lines)
    # roll_width = np.array(21) # (N+1) seats
    demand_width_array = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]) # [1,3,5,6] people in each group
    value_array = demand_width_array -1
    demand_number_array = np.array([170, 200, 360, 1000, 130, 720, 100, 800, 737, 570, 800, 630, 86, 351, 95, 100])
    start = time.time()

    a,b = IP_formulation(16, given_lines)
    generate_assignment(b)
    print("IP took...", round(time.time() - start, 2), "seconds")


