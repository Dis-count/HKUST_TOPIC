import gurobipy as grb
from gurobipy import GRB
import numpy as np
from collections import Counter
from SamplingMethod import samplingmethod
from Mist import generate_sequence, decision1

# This function uses IP to solve stochastic Model directly.

class originalModel:
    def __init__(self, roll_width, given_lines, demand_width_array, num_sample, I, prop, dw):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array - 1
        self.W = len(prop)
        self.I = I
        self.dw = dw
        self.prop = prop
        seat = np.insert(demand_width_array, 0, 1)
        self.seat_value = np.diff(seat)
        self.num_sample = num_sample

    def Wmatrix(self):
        # n is the dimension of group types
        return - np.identity(self.I) + np.eye(self.I, k=1)

    def solveModelGurobi(self):
        self.prop = self.prop * self.num_sample
        m2 = grb.Model()
        x = m2.addVars(self.I, self.given_lines, lb=0,
                       vtype=GRB.INTEGER, name='varx')
        y1 = m2.addVars(self.I, self.W, lb=0,  vtype=GRB.CONTINUOUS)
        y2 = m2.addVars(self.I, self.W, lb=0,  vtype=GRB.CONTINUOUS)
        W0 = self.Wmatrix()
        m2.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                   for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        M_identity = np.identity(self.I)

        m2.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) + grb.quicksum(W0[i, j] * y1[j, w] +
                      M_identity[i, j]*y2[j, w] for j in range(self.I)) == self.dw[w][i] for i in range(self.I) for w in range(self.W))
        # print("Constructing second took...", round(time.time() - start, 2), "seconds")
        m2.setObjective(grb.quicksum(self.num_sample* self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) - grb.quicksum(
            self.seat_value[i]*y1[i, w]*self.prop[w] for i in range(self.I) for w in range(self.W)), GRB.MAXIMIZE)

        # m2.setParam('OutputFlag', 0)

        m2.optimize()
        # print('optimal value:', m2.objVal)
        sol = np.array(m2.getAttr('X'))
        solx = sol[0:self.I * self.given_lines]
        newx = np.reshape(solx, (self.I, self.given_lines))
        # print('each row:', newx)
        newd = np.sum(newx, axis=1)
        # print('optimal demand:', newd)
        return newd, m2.objVal/self.num_sample

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    number_period = 60
    given_lines = 8
    # np.random.seed(0)

    probab = [0.4, 0.4, 0.1, 0.1]
    sam = samplingmethod(I, num_sample, number_period, probab)

    dw, prop = sam.get_prob()
    W = len(dw)

    # roll_width = np.arange(40, 40 + given_lines)
    roll_width = np.ones(given_lines) * 20

    demand_width_array = np.arange(2, 2+I)

    sequence = generate_sequence(number_period, probab)

    my = originalModel(roll_width, given_lines,
                         demand_width_array, num_sample, I, prop, dw)

    ini_demand, upperbound = my.solveModelGurobi()

    decision_list = decision1(sequence, ini_demand, probab)
    sequence = [i-1 for i in sequence if i > 0]
    total_people = np.dot(sequence, decision_list)
    final_demand = np.array(sequence) * np.array(decision_list)

    print(total_people)
    print(Counter(final_demand))
