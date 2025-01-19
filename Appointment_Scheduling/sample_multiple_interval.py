import gurobipy as grb
from gurobipy import GRB
import numpy as np

class originalModel:
    def __init__(self, zeta, people_num, num_sample, alpha):
        self.zeta = zeta
        self.people_num = people_num
        self.num_sample = num_sample
        self.alpha = alpha

    def solveModel(self):
        m2 = grb.Model()
        A = m2.addVars(self.people_num, lb = 0, vtype = GRB.CONTINUOUS, name = 'varA')
        S = m2.addVars(self.people_num, self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name= 'varS')
        Z = m2.addVars(self.people_num, self.people_num, self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name='varZ')
        O = m2.addVars(self.people_num, self.people_num, self.num_sample, lb = -float('inf'), ub=0, vtype=GRB.CONTINUOUS, name='varO')
        T = m2.addVars(self.people_num, self.people_num,
                       self.num_sample, lb = -float('inf'), vtype=GRB.CONTINUOUS, name='varT')

        m2.addConstrs(S[i, k] >= S[i-1, k] + self.zeta[i-1, k] for i in range(1, self.people_num) for k in range(self.num_sample))
          
        m2.addConstrs(S[i, k] >= A[i] for i in range(self.people_num) for k in range(self.num_sample))

        m2.addConstrs(Z[i, j, k] >= O[i, j, k] + A[j] - A[j-1] for i in range(self.people_num-2)
                       for j in range(i+2, self.people_num) for k in range(self.num_sample))

        # m2.addConstrs(O[i, j, k] <= S[i, k] - A[j] for i in range(self.people_num-2) for j in range(i+2, self.people_num) for k in range(self.num_sample))

        m2.addConstr(A[0] == 0)
        m2.addConstrs(T[i, j, k] == S[i, k]- A[j] for i in range(self.people_num-2)
                      for j in range(i+2, self.people_num) for k in range(self.num_sample))

        for i in range(self.people_num-2):
            for j in range(i+2, self.people_num):
                for k in range(self.num_sample):
                    m2.addConstr(O[i, j, k] == grb.min_([T[i, j, k]], constant=0))
                    # m2.addGenConstrMin(O[i, j, k], [S[i, k]-A[j]], 0)

        term0 = grb.quicksum(S[self.people_num-1, k] for k in range(self.num_sample))

        term1 = grb.quicksum(self.alpha[i, j] * Z[i, j, k] for i in range(1, self.people_num-2)
                             for j in range(i+2, self.people_num) for k in range(self.num_sample))

        m2.setObjective(term0 + term1, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        m2.write('1.lp')
        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        solx = sol[0: self.people_num]
        print('Solution:', solx)
        print(np.diff(solx))
        return solx

if __name__ == "__main__":
    num_sample = 10
    people_num = 3
    alpha = np.ones([people_num, people_num])
    zeta = np.random.randint(10, 20, [people_num, num_sample])
    # zeta = np.random.poisson(40, [people_num, num_sample])

    # zeta = np.random.exponential(45, [people_num, num_sample])
    appointment = originalModel(zeta, people_num, num_sample, alpha)

    solx = appointment.solveModel()