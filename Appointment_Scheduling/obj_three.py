import gurobipy as grb
from gurobipy import GRB
import numpy as np
from scipy.stats import truncnorm

# This function is used to compare the situations where only individual waiting, all constraints, without individual waiting are considered.

# Use SSA to simulate the problem

class originalModel:
    def __init__(self, people_num, num_sample, zeta, overlapping_thres, T, c_i, c_w, c_o):
        self.num_sample = num_sample
        self.people_num = people_num
        # the stochastic service time
        self.zeta = zeta
        self.wij = overlapping_thres
        self.w = overlapping_thres[0]
        self.T = T
        self.c_i = c_i
        self.c_w = c_w
        self.c_o = c_o

    def waiting(self):
        m2 = grb.Model()
        # Appointment interval
        delta = m2.addVars(self.people_num-1, lb = 0, vtype = GRB.CONTINUOUS, name = 'delta')
        over = m2.addVars(self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name='over')
        
        # Waiting time
        W = m2.addVars(self.people_num, self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name = 'W')
        
        term1 = grb.quicksum(delta[i] for i in range(self.people_num-1))
        m2.addConstrs(over[j] >= term1 + W[self.people_num-1, j] + self.zeta[self.people_num-1, j] - self.T for j in range(self.num_sample))

        # Waiting time for the first is 0
        m2.addConstrs(W[0, j] == 0 for j in range(self.num_sample))

        m2.addConstr(term1 <= self.T)

        m2.addConstrs(W[i, j] >= W[i-1, j] + self.zeta[i-1, j] - delta[i-1]
                      for i in range(1, self.people_num) for j in range(self.num_sample))

        m2.addConstrs((grb.quicksum(W[i, j] for j in range(self.num_sample))/self.num_sample <= self.w) for i in range(1, self.people_num))

        # waiting cost
        term2 = grb.quicksum(W[i, j] for i in range(self.people_num) for j in range(self.num_sample))/self.num_sample
        
        # overtime cost
        term3 = grb.quicksum(over[j] for j in range(self.num_sample))/self.num_sample
        term4 = grb.quicksum(self.zeta[i, j] for i in range(self.people_num-1) for j in range(self.num_sample))/self.num_sample

        # idle time cost
        term0 = term1 + grb.quicksum(W[self.people_num-1, j] for j in range(self.num_sample))/self.num_sample - term4

        m2.setObjective(self.c_w * term2 + self.c_i * term0 + self.c_o * term3, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()

        sol = np.array(m2.getAttr('X'))
        opt_delta = sol[0: self.people_num-1]
        print('Appointment Interval:', opt_delta)
        print('Appointment Time:', np.cumsum(opt_delta))
        
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

        print(m2.ObjVal)
        return delta

    def overlapping(self):
        # without single waiting
        m2 = grb.Model()

        # Appointment interval
        delta = m2.addVars(self.people_num-1, lb = 0, vtype = GRB.CONTINUOUS, name = 'delta')

        over = m2.addVars(self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name='over')

        W_ij_num = []
        for k in range(self.num_sample):
            for i in range(self.people_num):
                for j in range(i, self.people_num):
                    W_ij_num.append((i, j, j-i+1, k))

        # Waiting time for the first is 0
        w_ij = m2.addVars(grb.tuplelist(W_ij_num), lb = 0, vtype = GRB.CONTINUOUS, name = 'w_ij')

        term1 = grb.quicksum(delta[i] for i in range(self.people_num-1))

        m2.addConstr(term1 <= self.T)
        m2.addConstrs(over[j] >= term1 + w_ij[self.people_num-1, self.people_num-1, 1, j] +
                      self.zeta[self.people_num-1, j] - self.T for j in range(self.num_sample))

        m2.addConstrs(w_ij[i, j, j-i+1, k] >= w_ij[i-1, i-1, 1, k] + self.zeta[i-1, k] - (grb.quicksum(delta[i0] for i0 in range(i-1, j))) for i in range(1, self.people_num) for j in range(i, self.people_num) for k in range(self.num_sample))

        m2.addConstrs((grb.quicksum(w_ij[i, j, j-i+1, k] for k in range(self.num_sample))/self.num_sample <= self.wij[j-i]) for i in range(1, self.people_num-1) for j in range(i+1, min(i+2, self.people_num)))

        # waiting cost
        term2 = grb.quicksum(w_ij[i, i, 1, j] for i in range(self.people_num)
                             for j in range(self.num_sample))/self.num_sample

        # overtime cost
        term3 = grb.quicksum(over[j] for j in range(self.num_sample))/self.num_sample

        term4 = grb.quicksum(self.zeta[i, j] for i in range(self.people_num-1) for j in range(self.num_sample))/self.num_sample

        # idle time cost
        term0 = term1 + grb.quicksum(w_ij[self.people_num-1, self.people_num-1, 1, j]
                                     for j in range(self.num_sample))/self.num_sample - term4

        m2.setObjective(self.c_w * term2 + self.c_i * term0 + self.c_o * term3, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()

        if m2.Status == GRB.Status.OPTIMAL:
            sol = np.array(m2.getAttr('X'))
            delta = sol[0: self.people_num-1]
            print('Appointment Interval:', delta)
            print('Appointment Time:', np.cumsum(delta))

            total_waiting = 0
            for i in range(self.people_num):
                mean_i = sum(w_ij[i,i,1,k].X for k in range(self.num_sample))/self.num_sample
                total_waiting += mean_i
                print(f'Expected Waiting Time for {i+1}: {mean_i}')
            print(f'Total Waiting Time: {total_waiting}')

            # mean_ij = np.zeros(self.people_num)

            # for i in range(1, self.people_num-1):
            #     mean_ij[i] = sum(w_ij[i, i+1, 2, k].X for k in range(self.num_sample))/self.num_sample
            
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
        
        else:
            print('There is no optimal solution.')
            print(m2.Status)

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

        m2.addConstrs((grb.quicksum(w_ij[i, j, j-i+1, k] for k in range(self.num_sample))/self.num_sample <= self.wij[j-i]) for i in range(1, self.people_num) for j in range(i, self.people_num))

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
    people_num = 10
    overlapping_thres = np.zeros(people_num)
    overlapping_thres[0] = 50  # one person waiting
    overlapping_thres[1] = 20  # two person waiting
    overlapping_thres[2] = 5
    T = 200
    c_i = 5
    c_w = 1
    c_o = 0

    np.random.seed(0)

    # test normal distribution
    mu, sigma = 20, 20
    a, b = (0 - mu) / sigma, np.inf

    zeta = truncnorm.rvs(a, b, loc = mu, scale = sigma, size = (people_num, num_sample))

    # zeta = np.random.exponential(20, [people_num, num_sample])
    zeta = np.floor(zeta).astype(int)
    # zeta = np.clip(zeta, None, 60)

    appointment = originalModel(people_num, num_sample, zeta, overlapping_thres, T, c_i, c_w, c_o)

    # delta = appointment.waiting()

    delta1 = appointment.overlapping()

    # delta3 = appointment.all_waiting()


# 12. 23. 26. 26. 26. 25. 24. 18.  0.

# 6. 20. 23. 25  26.8 25.2 27 28. 19.     

# 公共卫生经验总结
