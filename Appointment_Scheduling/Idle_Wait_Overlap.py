# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:08:45 2024

@author: wliubi
"""

from gurobipy import *
import gurobipy as grb
import numpy as np


class Model:
    def __init__(self, zeta, people_num, num_sample, alpha, ddl):
        self.num_sample = num_sample
        self.people_num = people_num
        #### the stochastic service time
        self.zeta = zeta
        self.ddl = ddl
        #### the penalty cost for overlapping with different numbers of customers
        self.alpha = alpha

        
    def solveModel(self):
        m = grb.Model()
        #### the decision variables: scheduled arrival interval
        # x = m.addVars(self.people_num-1, lb = 0, vtype = GRB.CONTINUOUS, name = 'x')
        #x_actual = m.addVars(self.people_num-1, self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name = 'varx')
        #### the auxillary variables, indicating the overlapping time in each sample
        #### the arrival time
        A = m.addVars(self.people_num, lb = 0, vtype = GRB.CONTINUOUS, name = 'A')
        #### the service starting time
        S = m.addVars(self.people_num, self.num_sample, lb = 0, vtype = GRB.CONTINUOUS, name = 'S')
        #### the overlapping time
        equ_ij_num = []
        for k in range(self.num_sample):                    
            for i in range(self.people_num):
                for j in range(i, self.people_num):
                    equ_ij_num.append((i,j,j-i+1,k))

        w_ij_prime = m.addVars(tuplelist(equ_ij_num), lb=0, vtype = GRB.CONTINUOUS, name='w_ij_prime')
        #### auxillary decision variables for calculating the overlapping time
        w_ij = m.addVars(tuplelist(equ_ij_num), lb = 0, vtype = GRB.CONTINUOUS, name = 'w_ij')
        T = m.addVars(tuplelist(equ_ij_num), lb = 0, vtype = GRB.CONTINUOUS, name = 'varT')
        #### the constraints: 
        #### the uncertain arrival interval
        #m.addConstrs(x_actual[i, k] == x[i] * (1 + np.random.uniform(-0.1,0.3))
        #             for i in range(self.people_num-1) for k in range(self.num_sample))
        
        #### the n-th customer should be scheduled to arrive within the service time.
        m.addConstr(A[self.people_num-1] <= self.ddl)
        m.addConstr(A[0] == 0)
        #### calculating the arrival time  
        for k in range(self.num_sample):
            #### calculating the service starting time
            m.addConstrs(S[i, k] >= A[i] for i in range(self.people_num))
            m.addConstrs(S[i, k] >= S[i-1, k] + self.zeta[i-1, k] for i in range(1, self.people_num))
            
            m.addConstrs(T[i, j, j-i+1, k] == S[i, k] - A[j] for i in range(self.people_num)
                          for j in range(i, self.people_num))
            #### calculating the waiting and overlapping time
            m.addConstrs(w_ij[i, j, j-i+1, k] == grb.max_([T[i, j, j-i+1, k]], 0) for i in range(self.people_num) for j in range(i, self.people_num))

            m.addConstrs(w_ij[i, j, j-i+1, k] == grb.quicksum(w_ij_prime[t, l, l-t+1, k] for t in range(i+1) for l in range(j, self.people_num)) for i in range(self.people_num) for j in range(i, self.people_num))

        #### the objective
        #### the total idleness time
        idle_time = grb.quicksum(S[self.people_num-1, k] for k in range(self.num_sample))
        #### the total waitting + overlapping time
        overlap_time = {}
        for i in range(self.people_num):
            for j in range(i, self.people_num):
                overlap_time[j-i+1] = 0
        for k in range(self.num_sample):
            for i in range(self.people_num):
                for j in range(i, self.people_num):
                    overlap_time[j-i+1] = overlap_time[j-i+1] + self.alpha[j-i+1] * w_ij_prime[i, j, j-i+1, k]
       
        total_overlap_time = sum(overlap_time.values())

        m.setObjective(idle_time + total_overlap_time, GRB.MINIMIZE)

        m.setParam('OutputFlag', 0)
        m.write('2.lp')
        m.optimize()

        sol = np.array(m.getAttr('X'))
        solx = sol[0: self.people_num]
        print('Solution:', solx)
        print('The optimal objective is %g' % m.objVal)
        
        #### find the resulting idle time
        idle_t = {}
        for k in range(self.num_sample):
            idle_t[k] = S[self.people_num-1, k].x - np.sum(self.zeta[0:self.people_num-1, k])
        print('The idleness time is:', S[self.people_num-1, 0].x)
        
        for k in range(self.num_sample):
            for i in range(self.people_num):
                for j in range(i, self.people_num):
                    print('T:', T[i, j, j-i+1, k].x)

        #### find the resulting waiting time        
        # overlap_t = {}
        # for k in range(self.num_sample):
        #     overlap_t[k] = 0
        #     for i in range(self.people_num):
        #         for j in range(i, self.people_num):
        #             if t[i, j, j-i+1, k].x > 0.5:
        #                 print((i, j, j-i+1, k), t[i, j, j-i+1, k].x)
#                        overlap_t[j-i+1, k] = overlap_t[j-i+1, k] + t[i, j, j-i+1, k].x

        return [solx, idle_t]

if __name__ == "__main__":
    num_sample = 1
    people_num = 3
    alpha = {}
    penalty = 1
    for num in range(1,people_num+1):
        alpha[num] = penalty
        penalty = penalty + 0.1
    np.random.seed(20)
    zeta = np.trunc(np.random.exponential(10, [people_num, num_sample]))
    ddl = people_num * 10
    appointment = Model(zeta, people_num, num_sample, alpha, ddl)

    [solx, idle_t] = appointment.solveModel()
    
    
    