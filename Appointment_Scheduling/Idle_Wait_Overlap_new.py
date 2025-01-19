# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:08:45 2024

@author: wliubi
"""

from gurobipy import *
import gurobipy as grb
import numpy as np


class Model:
    def __init__(self, zeta, people_num, num_sample, alpha, ddl, alpha_original): 
        self.num_sample = num_sample
        self.people_num = people_num
        #### the stochastic service time
        self.zeta = zeta
        self.ddl = ddl
        #### the penalty cost for overlapping with different numbers of customers
        self.alpha = alpha
        self.alpha_original = alpha_original
        
    def solveModel(self):
        m = grb.Model()
        #### the decision variables: scheduled arrival interval
        # x = m.addVars(self.people_num-1, lb = 0, vtype = GRB.INTEGER, name = 'varx')
        #x_actual = m.addVars(self.people_num-1, self.num_sample, lb = 0, vtype = GRB.INTEGER, name = 'varx')
        #### the auxillary variables, indicating the overlapping time in each sample
        #### the arrival time
        A = m.addVars(self.people_num, lb = 0, vtype = GRB.CONTINUOUS, name = 'A')
        #### the service starting time
        S = m.addVars(self.people_num, self.num_sample, lb = 0,
                      vtype = GRB.CONTINUOUS, name = 'S')
        #### the overlapping time
        equ_ij_num = []
        for k in range(self.num_sample):                    
            for i in range(self.people_num):
                for j in range(i, self.people_num):
                    equ_ij_num.append((i, j, j-i+1, k))
                    
        w_ij_prime = m.addVars(tuplelist(equ_ij_num), lb = 0, vtype = GRB.CONTINUOUS, name = 'w_ij_prime')

        # add binary variables to linearize the formulation
        M_ij = m.addVars(tuplelist(equ_ij_num), vtype = GRB.BINARY, name='M_ij')
        #### auxillary decision variables for calculating the overlapping time
        w_ij = m.addVars(tuplelist(equ_ij_num), lb = 0, vtype = GRB.CONTINUOUS, name = 'w_ij')

        #### the constraints: 
        #### the uncertain arrival interval
        #m.addConstrs(x_actual[i, k] == x[i] * (1 + np.random.uniform(-0.1,0.3))
        #             for i in range(self.people_num-1) for k in range(self.num_sample))

        #### the n-th customer should be scheduled to arrive within the service time.
        m.addConstr(A[self.people_num-1] <= self.ddl)
        m.addConstr(A[0] == 0)

        m.addConstrs(A[i] <= A[i+1] for i in range(self.people_num-1))
        #### calculating the arrival time
        for k in range(self.num_sample):
            #### calculating the service starting time
            m.addConstrs(S[i, k] >= A[i] for i in range(self.people_num))
            m.addConstrs(S[i, k] >= S[i-1, k] + self.zeta[i-1, k] for i in range(1, self.people_num))
            
            #### calculating the waiting and overlapping time??????????????????
            m.addConstrs(w_ij[i, j, j-i+1, k] >= S[i, k] - A[j] for i in range(self.people_num) for j in range(i, self.people_num))

            m.addConstrs(w_ij[i, j, j-i+1, k] <= (S[i, k] - A[j]) + M_ij[i, j, j-i+1, k] * M
                         for i in range(self.people_num) for j in range(i, self.people_num))
            
            m.addConstrs(w_ij[i, j, j-i+1, k] <= (1- M_ij[i, j, j-i+1, k]) * M for i in range(self.people_num) for j in range(i, self.people_num))
            # m.addConstrs(w_ij_prime[i, self.people_num, self.people_num -i+1, k] == 0 for i in range(self.people_num))
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

        # total_overlap_time = 0
        # for key in overlap_time.keys():
        #     total_overlap_time = total_overlap_time + overlap_time[key]
        total_overlap_time = sum(overlap_time.values())

        m.setObjective(idle_time + total_overlap_time, GRB.MINIMIZE)

        m.setParam('OutputFlag', 0)
        # m.write('1.lp')
        m.optimize()

        sol = np.array(m.getAttr('X'))
        solx = sol[0: self.people_num]
        # print('Solution:', solx)
        # print('Interval:', np.diff(solx))

        # print('The optimal objective is %g' % m.objVal)
        # print('Overlap:', S[self.people_num-1, 0].x)
        #### find the resulting idle time
        idle_t = {}
        for k in range(self.num_sample):
            idle_t[k] = S[self.people_num-1, k].x - np.sum(self.zeta[0:self.people_num-1, k])

        # print('The idleness time is:', idle_t)
        arrival_t = {}
        start_t = {}
        for i in range(self.people_num):
            arrival_t[i] = A[i].x
            for k in range(self.num_sample):
                start_t[i, k] = S[i, k].x
        #print('The idleness time is:', idle_t)
        
#       #### find the resulting waiting time        
        # overlap_t = {}
        # for k in range(self.num_sample):
        #     overlap_t[k] = 0
        #     for i in range(self.people_num):
        #         for j in range(i, self.people_num):
        #             if  w_ij[i, j, j-i+1, k].x > 0.5:
        #                 print('duration_ij:', (i, j, j-i+1, k), w_ij[i, j, j-i+1, k].x)
        #                 print('start time of i:', S[i, k].x)#,
        #                 print('arrival time of j:', A[j].x)

                    # if  w_ij_prime[i, j, j-i+1, k].x > 0.5:
                        # print('duration_only_ij', (i, j, j-i+1, k), w_ij_prime[i, j, j-i+1, k].x)#, 
                            #   (S[i, k].x, A[j, k].x), (S[i-1, k].x, A[j, k].x), (S[i, k].x, A[j+1, k].x)
                        # overlap_t[j-i+1, k] = overlap_t[j-i+1, k] + t[i, j, j-i+1, k].x

        return [solx, arrival_t, start_t]

    def originalModel(self):
        m2 = grb.Model()
        A = m2.addVars(self.people_num, lb = 0,
                       vtype = GRB.CONTINUOUS, name='varA')
        S = m2.addVars(self.people_num, self.num_sample, lb = 0,
                       vtype = GRB.CONTINUOUS, name='varS')

        m2.addConstrs(S[i, k] >= S[i-1, k] + self.zeta[i-1, k]
                      for i in range(1, self.people_num) for k in range(self.num_sample))

        m2.addConstrs(A[i] <= A[i+1] for i in range(self.people_num-1))

        m2.addConstrs(S[i, k] >= A[i] for i in range(self.people_num)
                      for k in range(self.num_sample))
        m2.addConstr(A[self.people_num-1] <= self.ddl)

        m2.addConstr(A[0] == 0)

        term0 = grb.quicksum(S[self.people_num-1, k]
                             for k in range(self.num_sample))

        term1 = grb.quicksum(self.alpha_original[i] * (S[i, k] - A[i])
                             for i in range(1, self.people_num) for k in range(self.num_sample))

        m2.setObjective(term0 + term1, GRB.MINIMIZE)
        m2.setParam('OutputFlag', 0)
        # m2.write('1.lp')
        m2.optimize()
        # print('The optimal objective is %g' % m2.objVal)
        for i in range(self.people_num):
            print('Original starting time:', S[i, 0].x)
        sol = np.array(m2.getAttr('X'))
        solx = sol[0: self.people_num]
        print('Solution:', solx)
        print('Original:', np.diff(solx))

        return solx

if __name__ == "__main__":
    num_sample = 500
    people_num = 16
    M = 1000
    alpha = {}
    alpha_1 = {}
    penalty = 1
    ddl = people_num * 30

    for num in range(1,people_num+1):
        alpha[num] = penalty
        penalty += 4

    for num in range(1, people_num+1):
        alpha_1[num] = 1
    np.random.seed(20)
    zeta = np.trunc(np.random.exponential(30, [people_num, num_sample]))
    # print(zeta)    
    appointment = Model(zeta, people_num, num_sample, alpha, ddl, alpha_1)
    appointment.originalModel()
    # [solx, arrival_t, start_t] = appointment.solveModel()

    # print(arrival_t)
    # print(start_t)
    
    