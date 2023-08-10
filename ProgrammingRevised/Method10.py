import gurobipy as grb
from gurobipy import GRB
import numpy as np
import copy
from SamplingMethod import samplingmethod
from Mist import generate_sequence, decision1
from collections import Counter
from Mist import decisionSeveral, decisionOnce
from Method1 import stochasticModel

# This function uses deterministicModel to make several decisions with initial stochastic solution.

class deterministicModel:
    def __init__(self, roll_width, given_lines, demand_width_array, I):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array - 1
        self.I = I
        self.s = 1

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
        # m.write('test.lp')
        m.optimize()
        if m.status !=2:
            m.write('test.lp')
        # print('************************************************')
        # print('Optimal value of IP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newx = np.rint(newx)
        newd = np.sum(newx, axis=1)
        newd = np.rint(newd)
        return newd, newx

    def IP_formulation2(self, roll_width, num_period, probab, seq):
        demand = np.ceil(np.array(probab) * num_period)
        demand[seq-2] +=1
        m = grb.Model()

        x = m.addVars(self.I, len(roll_width), lb=0, vtype= GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= roll_width[j] for j in range(len(roll_width)))
        m.addConstrs(grb.quicksum(x[i, j] for j in range(
            len(roll_width))) <= demand[i] for i in range(self.I))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(
            self.I) for j in range(len(roll_width))), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        # print('************************************************')
        # print('Optimal value of IP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        
        newx = np.reshape(x_ij, (self.I, len(roll_width)))
        newd = np.sum(newx, axis=1)
        return newd, newx

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

    def LP_formulation2(self, demand, roll_width):
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
        y_ij = np.array(m.getAttr('X'))[:self.I]
        return x_ij, y_ij, m.objVal

    def IP_formulation1(self, demand_lower, demand_upper):
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
            x_ij = np.array(m.getAttr('X'))
            newx = np.reshape(x_ij, (self.I, self.given_lines))
            return True
        else:
            return False

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    number_period = 80
    given_lines = 8
    np.random.seed(0)

    probab = [0.4, 0.4, 0.1, 0.1]
    sam = samplingmethod(I, num_sample, number_period, probab)

    dw, prop = sam.get_prob()
    W = len(dw)

    roll_width = np.arange(21, 21 + given_lines)
    total_seat = np.sum(roll_width)

    demand_width_array = np.arange(2, 2+I)

    sequence = generate_sequence(number_period, probab)
    sequence1 = copy.deepcopy(sequence)
    total_usedDemand = np.zeros(I)
    ini_demand1 = np.array(probab) * number_period

    my = stochasticModel(roll_width, given_lines,
                         demand_width_array, W, I, prop, dw)

    ini_demand, upperbound = my.solveBenders(eps=1e-4, maxit=20)

    mylist = []
    remaining_period0 = number_period
    deterModel = deterministicModel(
        roll_width, given_lines, demand_width_array, I)
        
    while remaining_period0:
        demand = ini_demand - total_usedDemand

        usedDemand, remaining_period = decisionSeveral(sequence, demand)

        diff_period = remaining_period0 - remaining_period

        mylist += [1] * diff_period

        if any(usedDemand) == 0:  # all are 0
            usedDemand, decision_list = decisionOnce(sequence, demand, probab)
            if decision_list:
                mylist.append(1)
            else:
                mylist.append(0)
            remaining_period -= 1

        remaining_period0 = remaining_period
        sequence = sequence[-remaining_period:]

        total_usedDemand += usedDemand

        ini_demand1 = total_usedDemand + \
            np.ceil(np.array(probab) * remaining_period)


        ini_demand, obj = deterModel.IP_formulation(total_usedDemand, ini_demand1)

    sequence1 = [i-1 for i in sequence1 if i > 0]
    total_people1 = np.dot(sequence1, mylist)
    final_demand1 = np.array(sequence1) * np.array(mylist)

    print(f'The number of seats: {total_seat}')
    print(f'The number of people:{total_people1}')
    print(Counter(final_demand1))

