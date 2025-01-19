import gurobipy as grb
from gurobipy import GRB
import numpy as np

class originalModel:
    def __init__(self, zeta, people_num, num_sample, alpha, beta):
        self.alpha = alpha
        self.num_sample = num_sample
        self.people_num = people_num
        self.zeta = zeta
        self.beta = beta

    def solveModel(self):
        m2 = grb.Model()
        x = m2.addVars(self.people_num-1, lb = 0, vtype = GRB.CONTINUOUS, name = 'varx')
        K = m2.addVars(self.people_num, self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name= 'vark')
        
        m2.addConstrs(K[0, j] == 0 for j in range(self.num_sample))

        m2.addConstrs(K[i, j] >= K[i-1, j] + self.zeta[i-1, j] - x[i-1]
                      for i in range(1, self.people_num) for j in range(self.num_sample))

        term1 = grb.quicksum((1 + self.beta)*x[i] for i in range(self.people_num-1))
        term2 = grb.quicksum(self.alpha[i] * K[i, j] for j in range(self.num_sample)
                             for i in range(1, self.people_num))

        m2.setObjective(term1 + term2/self.num_sample, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        solx = sol[0: self.people_num-1]
        print('Solution:', solx)

        return solx

if __name__ == "__main__":
    num_sample = 100
    people_num = 10
    alpha = [i+1 for i in range(people_num)]
    beta = 0
    alpha[-1] += (1+beta)
    # zeta = np.random.randint(30, 60, [people_num, num_sample])
    # zeta = np.random.poisson(40, [people_num, num_sample])

    zeta = np.random.exponential(30, [people_num, num_sample])
    appointment = originalModel(zeta, people_num, num_sample, alpha, beta)

    solx = appointment.solveModel()