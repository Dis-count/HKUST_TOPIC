import gurobipy as grb
from gurobipy import GRB
import numpy as np
from scipy.stats import truncnorm

# This function is used to compare the situations where only individual waiting, all constraints, without individual waiting are considered.

# Use SSA to simulate the problem

class originalModel:
    def __init__(self, people_num, num_sample, zeta, overlapping_thres):
        self.num_sample = num_sample
        self.people_num = people_num
        # the stochastic service time
        self.zeta = zeta
        self.wij = overlapping_thres
        self.w = overlapping_thres[0]

    def waiting(self):
        m2 = grb.Model()
        # Appointment interval
        delta = m2.addVars(self.people_num-1, lb = 0, vtype = GRB.CONTINUOUS, name = 'delta')
                          
        # Waiting time
        W = m2.addVars(self.people_num, self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name = 'W')
        
        # Waiting time for the first is 0
        m2.addConstrs(W[0, j] == 0 for j in range(self.num_sample))

        m2.addConstrs(W[i, j] >= W[i-1, j] + self.zeta[i-1, j] - delta[i-1]
                      for i in range(1, self.people_num) for j in range(self.num_sample))

        m2.addConstrs((grb.quicksum(W[i, j] for j in range(self.num_sample))/self.num_sample <= self.w) for i in range(1, self.people_num))

        term1 = grb.quicksum(W[self.people_num-1, j] + self.zeta[self.people_num-1, j]
                             for j in range(self.num_sample))
        term2 = grb.quicksum(delta[i] for i in range(self.people_num-1))

        m2.setObjective(term1/self.num_sample + term2, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()
 
        sol = np.array(m2.getAttr('X'))
        opt_delta = sol[0: self.people_num-1]
        print('Appointment Interval:', opt_delta)
        # print('Appointment Time:', np.cumsum(opt_delta))
        
        total_waiting = 0
        for i in range(self.people_num):
            mean_i = sum(W[i, j].X for j in range(self.num_sample)) / self.num_sample
            total_waiting += mean_i
            print(f'Expected Waiting Time for {i+1}: {mean_i}')
        print(f'Total Waiting Time: {total_waiting}')

        mean_wij = np.zeros(self.people_num)
        for i in range(1, self.people_num-1):
            mean_w = 0
            delta_term = sum(opt_delta[i0] for i0 in range(i-1, i+1))

            for k in range(self.num_sample):
                val = W[i-1, k].X + self.zeta[i-1, k] - delta_term
                mean_w += max(val, 0)
            mean_wij[i] = mean_w/self.num_sample

        for i in range(1, self.people_num-1):
            print(f'Expected Overlap Time for {i+1} and {i+2}: {mean_wij[i]}')
        print(f'Total Overlap Time: {sum(mean_wij)}')
            # print(f'Expected Overlapping Time for {i+1}: {mean_wij[i]+mean_wij[i-1]}')

        # var = m2.getVarByName('W')
        # print(var.X)

        # expected_waiting = np.mean(var.X, axis=0)
        # print('Expected Waiting Time for Each:', expected_waiting)
        print(m2.ObjVal)
        return delta

    def overlapping(self):
        # without single waiting
        m2 = grb.Model()

        # Appointment interval
        delta = m2.addVars(self.people_num-1, lb = 0,
                           vtype = GRB.CONTINUOUS, name = 'delta')

        W_ij_num = []
        for k in range(self.num_sample):
            for i in range(self.people_num):
                for j in range(i, self.people_num):
                    W_ij_num.append((i, j, j-i+1, k))

        # Waiting time for the first is 0
        w_ij = m2.addVars(grb.tuplelist(W_ij_num), lb = 0, vtype = GRB.CONTINUOUS, name = 'w_ij')

        # m2.addConstr(delta[0] == 8)

        m2.addConstrs(w_ij[i, j, j-i+1, k] >= w_ij[i-1, i-1, 1, k] + self.zeta[i-1, k] - (grb.quicksum(delta[i0] for i0 in range(i-1, j))) for i in range(1, self.people_num) for j in range(i, self.people_num) for k in range(self.num_sample))

        m2.addConstrs((grb.quicksum(w_ij[i, j, j-i+1, k] for k in range(self.num_sample))/self.num_sample <= self.wij[j-i]) for i in range(1, self.people_num-1) for j in range(i+1, min(i+2, self.people_num)))

        term1 = grb.quicksum(w_ij[self.people_num-1, self.people_num-1, 1, k] + self.zeta[self.people_num-1, k] for k in range(self.num_sample))
        term2 = grb.quicksum(delta[i] for i in range(self.people_num-1))

        m2.setObjective(term1/self.num_sample + term2, GRB.MINIMIZE)
        # m2.setObjective(term2, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        delta = sol[0: self.people_num-1]
        print('Appointment Interval:', delta)
        print('Appointment Time:', np.cumsum(delta))

        total_waiting = 0
        for i in range(self.people_num):
            mean_i = sum(w_ij[i,i,1,k].X for k in range(self.num_sample)) / self.num_sample
            print(f'Expected Waiting Time for {i+1}: {mean_i}')
            total_waiting += mean_i
        print(f'Total Waiting Time: {total_waiting}')

        # mean_ij = np.zeros(self.people_num)

        # for i in range(1, self.people_num-1):
        #     mean_ij[i] = sum(w_ij[i, i+1, 2, k].X for k in range(self.num_sample)) / self.num_sample
        
        # for i in range(1, self.people_num-1):
        #     # print(f'Expected Overlapping Time for {i+1}: {mean_ij[i]+mean_ij[i-1]}')
        #     print(f'Expected Overlapping Time for {i+1} and {i+2}: {mean_ij[i]}')
        # print(f'Total Overlap Time: {sum(mean_ij)}')

        mean_wij = np.zeros(self.people_num)
        for i in range(1, self.people_num-1):
            mean_w = 0
            delta_term = sum(delta[i0] for i0 in range(i-1, i+1))

            for k in range(self.num_sample):
                val = w_ij[i-1, i-1, 1, k].X + self.zeta[i-1, k] - delta_term
                mean_w += max(val, 0)
            mean_wij[i] = mean_w/self.num_sample

        for i in range(1, self.people_num-1):
            print(f'Expected Overlap Time for {i+1} and {i+2}: {mean_wij[i]}')
        print(f'Total Overlap Time: {sum(mean_wij)}')

        print(m2.ObjVal)
        return delta

    def two_overlapping(self):
        # without single waiting
        # E23 + E34 \leq threshold
        m2 = grb.Model()

        # Appointment interval
        delta = m2.addVars(self.people_num-1, lb=0,
                           vtype=GRB.CONTINUOUS, name='delta')

        W_ij_num = []
        for k in range(self.num_sample):
            for i in range(self.people_num):
                for j in range(i, self.people_num):
                    W_ij_num.append((i, j, j-i+1, k))

        # Waiting time for the first is 0
        w_ij = m2.addVars(grb.tuplelist(W_ij_num), lb=0, vtype = GRB.CONTINUOUS, name='w_ij')

        # m2.addConstrs(w_ij[0, 0, 1, k] == 0 for k in range(self.num_sample))

        m2.addConstrs(w_ij[i, j, j-i+1, k] >= w_ij[i-1, i-1, 1, k] + self.zeta[i-1, k] - (grb.quicksum(delta[i0] for i0 in range(i-1, j))) for i in range(1, self.people_num) for j in range(i, self.people_num) for k in range(self.num_sample))

        # m2.addConstr(delta[0] == 0)
        # m2.addConstr(delta[1] == 36)

        m2.addConstrs((grb.quicksum(w_ij[i, i+1, 2, k] + w_ij[i-1, i, 2, k] for k in range(self.num_sample))/self.num_sample <= self.wij[1]) for i in range(1, self.people_num-1))

        term1 = grb.quicksum(w_ij[self.people_num-1, self.people_num-1, 1, k] +
                             self.zeta[self.people_num-1, k] for k in range(self.num_sample))
        term2 = grb.quicksum(delta[i] for i in range(self.people_num-1))

        m2.setObjective(term1/self.num_sample + term2, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        delta = sol[0: self.people_num-1]
        print('Appointment Interval:', delta)
        # print('Overlapping Appointment Time:', np.cumsum(delta))

        for i in range(self.people_num):
            mean_i = sum(w_ij[i, i, 1, k].X for k in range(
                self.num_sample)) / self.num_sample
            print(f'Expected Waiting Time for {i+1}: {mean_i}')

        mean_ij = np.zeros(self.people_num)

        for i in range(1, self.people_num-1):
            mean_ij[i] = sum(
                w_ij[i, i+1, 2, k].X for k in range(self.num_sample)) / self.num_sample

        for i in range(1, self.people_num):
            print(f'Expected Overlapping Time for {i+1}: {mean_ij[i]+mean_ij[i-1]}')

        print(mean_ij)
        print(m2.ObjVal)
        return delta

    def all_waiting(self):
        m2 = grb.Model()

        # Appointment interval
        delta = m2.addVars(self.people_num-1, lb=0,
                           vtype=GRB.CONTINUOUS, name='delta')
        # All waiting time

        W_ij_num = []
        for k in range(self.num_sample):
            for i in range(self.people_num):
                for j in range(i, self.people_num):
                    W_ij_num.append((i, j, j-i+1, k))

        # Waiting time for the first is 0
        w_ij = m2.addVars(grb.tuplelist(W_ij_num), lb=0,
                          vtype = GRB.CONTINUOUS, name='w_ij')

        # m2.addConstrs(w_ij[0, 0, 1, k] == 0 for k in range(self.num_sample))

        m2.addConstrs(w_ij[i, j, j-i+1, k] >= w_ij[i-1, i-1, 1, k] + self.zeta[i-1, k] - (grb.quicksum(delta[i0] for i0 in range(i-1, j))) for i in range(1, self.people_num) for j in range(i, self.people_num) for k in range(self.num_sample))

        m2.addConstrs((grb.quicksum(w_ij[i, j, j-i+1, k] for k in range(self.num_sample))/self.num_sample <= self.wij[j-i]) for i in range(1, self.people_num) for j in range(i, min(i+2, self.people_num)))

        term1 = grb.quicksum(w_ij[self.people_num-1, self.people_num-1, 1, k] +
                             self.zeta[self.people_num-1, k] for k in range(self.num_sample))
        term2 = grb.quicksum(delta[i] for i in range(self.people_num-1))

        m2.setObjective(term1/self.num_sample + term2, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        delta = sol[0: self.people_num-1]
        print('Appointment Interval:', delta)
        # print('Overlapping Appointment Time:', np.cumsum(delta))
        print(m2.ObjVal)

        return delta



if __name__ == "__main__":
    num_sample = 5000
    people_num = 6
    overlapping_thres = np.zeros(people_num)
    overlapping_thres[0] = 30  # one person waiting
    overlapping_thres[1] = 5  # two person waiting
    # overlapping_thres[2] = 5

    np.random.seed(0)

    # test normal distribution
    # mu, sigma = 30, 15
    # a, b = (0 - mu) / sigma, np.inf

    # zeta = truncnorm.rvs(a, b, loc = mu, scale = sigma, size = (people_num, num_sample))
     
    zeta = np.random.exponential(25, [people_num, num_sample])
    zeta = np.floor(zeta).astype(int)
    # zeta = np.clip(zeta, None, 60)

    appointment = originalModel(people_num, num_sample, zeta, overlapping_thres)

    # delta = appointment.waiting()

    delta1 = appointment.overlapping()

    # delta2 = appointment.two_overlapping()

    # delta3 = appointment.all_waiting()

