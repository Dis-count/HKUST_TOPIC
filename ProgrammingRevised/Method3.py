import gurobipy as grb
from gurobipy import GRB
import numpy as np
import copy
from SamplingMethod import samplingmethod
from Mist import generate_sequence, decision1
from collections import Counter
from Mist import decisionSeveral, decisionOnce

# This function uses deterministicModel to make several decisions with initial deterministic solution.

class deterministicModel:
    def __init__(self, roll_width, given_lines, demand_width_array, I):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array - 1
        self.I = I

    def IP_formulation(self, demand_lower, demand_upper):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype=GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        if sum(demand_upper)!= 0:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(
            self.given_lines)) <= demand_upper[i] for i in range(self.I))
        if sum(demand_lower)!= 0:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(
            self.given_lines)) >= demand_lower[i] for i in range(self.I))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(
            self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        # print('************************************************')
        # print('Optimal value of IP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)
        return newd, m.objVal


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

    deterModel = deterministicModel(roll_width, given_lines, demand_width_array, I)

    ini_demand, obj = deterModel.IP_formulation(total_usedDemand, ini_demand1)

    mylist = []
    remaining_period0 = number_period

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

        deterModel = deterministicModel(
            roll_width, given_lines, demand_width_array, I)

        ini_demand, obj = deterModel.IP_formulation(total_usedDemand, ini_demand1)

    sequence1 = [i-1 for i in sequence1 if i > 0]
    total_people1 = np.dot(sequence1, mylist)
    final_demand1 = np.array(sequence1) * np.array(mylist)

    print(f'The number of seats: {total_seat}')
    print(f'The number of people:{total_people1}')
    print(Counter(final_demand1))

