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
        x = m2.addVars(self.people_num-1, vtype = GRB.CONTINUOUS, name = 'varx')
        O = m2.addVars(self.people_num-1, self.num_sample, lb = -float('inf'), vtype=GRB.CONTINUOUS, name='varK')
        Z = m2.addVars(self.people_num-1, self.num_sample, lb=0,
                       vtype = GRB.CONTINUOUS, name='varZ')

        m2.addConstrs(O[0, j] == 0 for j in range(self.num_sample))

        m2.addConstrs(O[i, j] >= O[i-1, j] + self.zeta[i-1, j] - x[i]
                      for i in range(1, self.people_num-1) for j in range(self.num_sample))
        m2.addConstrs(O[i, j] >= - x[i]
                      for i in range(1, self.people_num-1) for j in range(self.num_sample))

        m2.addConstrs(Z[i, j] >= O[i, j] for i in range(1, self.people_num-1) for j in range(self.num_sample))

        # m2.addConstrs(Z[i, j] >= 0 for i in range(self.people_num-1) for j in range(self.num_sample))

        term1 = grb.quicksum(x[i] for i in range(self.people_num-1))
        term2 = grb.quicksum(self.alpha[i] * Z[i, j] for j in range(self.num_sample)
                             for i in range(1, self.people_num-1))
        term3 = grb.quicksum(O[self.people_num-2, j] for j in range(self.num_sample))

        m2.setObjective(term1 + term2/self.num_sample + term3/self.num_sample, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        solx = sol[0: self.people_num-1]
        print('Solution:', solx)

        return solx

if __name__ == "__main__":

    for i in range(100):
        num_sample = 5000
        people_num = 15
        alpha = [2] * people_num
        beta = 0
        zeta = np.random.randint(30, 60, [people_num, num_sample])
        # zeta = np.random.poisson(40, [people_num, num_sample])

        # zeta = np.random.exponential(45, [people_num, num_sample])

        appointment = originalModel(zeta, people_num, num_sample, alpha, beta)
        solx = appointment.solveModel()

# [0,  0.  63. 116. 165. 213. 260. 307. 353. 399. 445. 490. 535. 578. 681.]


# [0.  63.  53.  49.  48.  47.  47.  46.  46.  46.  45.  45.  43. 103.]