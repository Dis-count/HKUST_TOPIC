import gurobipy as grb
from gurobipy import GRB
import numpy as np

# Use A, S to solve the appointment scheduling

class originalModel:
    def __init__(self, zeta, people_num, num_sample, alpha, ddl):
        self.zeta = zeta
        self.people_num = people_num
        self.num_sample = num_sample
        self.alpha = alpha
        self.ddl = ddl

    def idle_waiting(self):
        m2 = grb.Model()
        A = m2.addVars(self.people_num, lb = 0, vtype = GRB.CONTINUOUS, name = 'varA')
        S = m2.addVars(self.people_num, self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name= 'varS')

        m2.addConstrs(S[i, k] >= S[i-1, k] + self.zeta[i-1, k] for i in range(1, self.people_num) for k in range(self.num_sample))
        
        m2.addConstrs(S[i, k] >= A[i] for i in range(self.people_num) for k in range(self.num_sample))
        m2.addConstr(A[self.people_num-1] <= self.ddl)

        # m2.addConstrs(O[i, j, k] <= S[i, k] - A[j] for i in range(self.people_num-2) for j in range(i+2, self.people_num) for k in range(self.num_sample))

        m2.addConstr(A[0] == 0)

        term0 = grb.quicksum(S[self.people_num-1, k] for k in range(self.num_sample))

        term1 = grb.quicksum(self.alpha[i] * (S[i, k]- A[i]) for i in range(1, self.people_num) for k in range(self.num_sample))

        m2.setObjective(term0 + term1, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()
        print('The optimal objective is %g' % m2.objVal)
        print(S[self.people_num-1, 0].x)
        sol = np.array(m2.getAttr('X'))
        solx = sol[0: self.people_num]
        # print('Solution:', solx)
        print(np.diff(solx))
        return solx

if __name__ == "__main__":
    num_sample = 1000
    people_num = 10
    # np.random.seed(10)
    alpha = np.ones(people_num)
    # zeta = np.random.randint(10, 20, [people_num, num_sample])
    # zeta = np.random.poisson(40, [people_num, num_sample])

    zeta = np.trunc(np.random.exponential(20, [people_num, num_sample]))
    # print(zeta)
    ddl = people_num * 20
    appointment = originalModel(zeta, people_num, num_sample, alpha, ddl)

    solx = appointment.idle_waiting()
