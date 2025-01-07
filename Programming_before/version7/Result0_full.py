import gurobipy as grb
from gurobipy import GRB
import numpy as np

# This function uses deterministicModel to make several decisions with initial deterministic solution.

class deterministicModel:
    def __init__(self, roll_width, given_lines, I, s):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.s = s
        self.I = I
        self.demand_width_array = np.arange(1+self.s, self.I+1+self.s)
        self.value_array = np.arange(1, self.I+1)

    def IP_formulation(self, demand_lower, demand_upper):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype= GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        if sum(demand_upper) != 0:
            m.addConstrs(grb.quicksum(x[i, j] for j in range(
                self.given_lines)) <= demand_upper[i] for i in range(self.I))
        if sum(demand_lower) != 0:
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
        return newd, newx

    def full(self, newx, demand):
        newd = np.sum(newx, axis=1)
        num_fullorlargest = 0
        num_non = 0
        if sum(newd) < sum(demand):
            

            r = self.roll_width % (self.s + self.I)
            q = np.floor(self.roll_width/(self.s + self.I))
            largest = [q[i] * self.I + max(r[i] - self.s, 0) for i in range(self.given_lines)]

            for i in range(self.given_lines):
                if np.dot(newx[:,i], np.arange(1, self.I+1)) < largest[i]:
                    
                    if np.dot(newx[:,i], np.arange(1+self.s, self.I+1+self.s)) < self.roll_width[i]:
                        num_non += 1

            num_fullorlargest = self.given_lines - num_non
        
        return num_non, num_fullorlargest


if __name__ == "__main__":
    I = 4  # the number of group types

    given_lines = 10

    roll_width = np.ones(given_lines) * 21
    
    s = 1
    # np.random.seed(20)
    total_non = 0
    total_fullorlargest = 0
    total_instance = 0

    for i in range(1000):
        demand = np.random.randint(14, 24, size =4)
        demand_lower = np.zeros(4)
        
        a = deterministicModel(roll_width, given_lines, I, s)

        newd, newx = a.IP_formulation(demand_lower, demand)
        
        num_non, num_fullorlargest = a.full(newx, demand)

        if num_non > 0:
            total_instance += 1

        total_non += num_non
        total_fullorlargest += num_fullorlargest
    print(total_non)
    print(total_fullorlargest)
    print(total_instance)

    # 148
    # 9752

    # 167
    # 9793
    # 167
