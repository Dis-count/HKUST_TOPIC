import gurobipy as grb
from gurobipy import GRB
import numpy as np
import time
from collections import Counter
from scipy.stats import binom
import copy
# This function is used to calculate the revised benders (without benders decomposition)
# For each sub-problem, solve IP instead. 
# Can be used to compare IP and original model

class stochasticModel:
    def __init__(self, roll_width, given_lines, demand_width_array, W,I, prop, dw):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array - 1
        self.W = W
        self.I = I
        self.dw = dw
        self.prop = prop

    def obtainY(self, ind_dw, d0):
        # the last element is dummy
        yplus = np.zeros(self.I+1)
        yminus = np.zeros(self.I+1)

        for j in range(self.I-1, -1, -1):
            if ind_dw[j] > (d0[j] + yplus[j+1]):
                yminus[j] = ind_dw[j] - d0[j] - yplus[j+1]
            else:
                yplus[j] = - ind_dw[j] + d0[j] + yplus[j+1]
        return yplus, yminus

    def solveSub(self, yplus, yminus):
        # the first element is dummy
        # Thus, we should notice that the positions of \alpha and y are different.
        alpha = np.zeros(self.I+1)
        for j in range(self.I):
            if yplus[j] > 0:
                alpha[j+1] = alpha[j] + 1
            elif yplus[j] == 0 and yminus[j] == 0:
                if yplus[j+1] == 0:
                    alpha[j+1] = alpha[j] + 1
        alpha0 = np.delete(alpha, 0).tolist()
        return alpha0

    # def setupMasterModel(self):
    #     # Initially, add z<= 0
    #     m = grb.Model()
    #     x = m.addVars(self.I, self.given_lines, lb=0, vtype= GRB.CONTINUOUS, name = 'varx')
    #     z = m.addVars(self.W, lb = -float('inf'), vtype= GRB.CONTINUOUS, name = 'varz')
    #     m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
    #                             for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
    #     for i in range(self.W):
    #         #     yplus1, yminus1 = obtainY(dw[i])
    #         #     alpha0 = solveSub(yplus1, yminus1)
    #         m.addConstr(z[i] <= 0)
    #     #     print('Initial cut:', alpha0)
    #     m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) + grb.quicksum(self.prop[w] * z[w] for w in range(self.W)), GRB.MAXIMIZE)
    #     # m.update()
    #     m.setParam('OutputFlag', 0)
    #     # m.write('Master.lp')
    #     m.optimize()

    #     return m, [m.getVarByName('varx[' + str(i) + ',' + str(j) + ']').x for i in range(self.I) for j in range(self.given_lines)], [m.getVarByName('varz[' + str(w) + ']').x for w in range(self.W)]

    #  This function is used to generate new constraints.
    def add_Constrs(self, xstar, zstar):
        # xstar is the [I* given_lines, 1] vector
        newx = np.reshape(xstar, (self.I, self.given_lines))
        d0 = np.sum(newx, axis=1)
        LB = self.value_array @ d0
        alphaset = []
        wset = []

        for w in range(self.W):
            yplus, yminus = self.obtainY(self.dw[w], d0)
            alpha = self.solveSub(yplus, yminus) # alpha is a list
            objectiveAlpha = np.dot(alpha, (self.dw[w] - d0))
            LB += self.prop[w] * objectiveAlpha
            # print('LB:', LB)
            # print('alpha:', objectiveAlpha)
            
            if objectiveAlpha < zstar[w]:
                # print('Add the cut:', alpha, 'the scenario is:', w+1)
                alphaset.append(alpha)
                wset = np.append(wset, w+1)
        alphaset = np.array(alphaset)
        # print('alphaset:', alphaset)
        # print('zstar:', zstar)
        if len(wset):
            wset = wset.astype(int)  # change to int

        return alphaset, wset, LB

    # Recall that column generation needs more variables, so it will be difficult to repeat the same model.
    # But for the common benders model, the variables will be confirmed at the beginning.

    # This function add new set to model m.
    def solve_LP(self, m, alphaset, wset):
        start = time.time()
        if len(wset)==0:
            return m, m.objVal, m.getAttr(GRB.Attr.X, m.getVars()[0: self.I * self.given_lines]), m.getAttr(GRB.Attr.X, m.getVars()[-self.W:])
        # print('wset:', len(wset))
        x_var = m.getVars()[0: self.I * self.given_lines]
        z_var = m.getVars()[-self.W:]

        for t, k in enumerate(wset):
            # m.addConstr(grb.quicksum(alphaset[t][i] * m.getVarByName('varx[' + str(i) + ',' + str(j) + ']') for i in range(self.I) for j in range(self.given_lines)) + m.getVarByName('varz[' + str(k-1) + ']') <= grb.quicksum(alphaset[t][i] * self.dw[k-1][i] for i in range(self.I)))
            m.addConstr(grb.quicksum(alphaset[t][i] * x_var[i*self.given_lines+j] for i in range(self.I) for j in range(self.given_lines)) + z_var[k-1] <= grb.quicksum(alphaset[t][i] * self.dw[k-1][i] for i in range(self.I)))
        # [m.getVarByName('varx[' + str(i) + ',' + str(j) + ']').x for i in range(self.I) for j in range(self.given_lines)]
        # [m.getVarByName('varz[' + str(w) + ']').x for w in range(self.W)]
        print("Constructing LP took...", round(time.time() - start, 3), "seconds")
        m.setParam('OutputFlag', 0)
        # m.write('Master1.lp')
        m.optimize()

        return m, m.objVal, m.getAttr(GRB.Attr.X, m.getVars()[0: self.I * self.given_lines]), m.getAttr(GRB.Attr.X, m.getVars()[-self.W:])

    def addDemand(self, m, demand):
        # This function is used to add demand constraints to the model.
        x_var = m.getVars()[0: self.I * self.given_lines]
        m.addConstrs(grb.quicksum(x_var[i*self.given_lines+j] for j in range(self.given_lines)) >= demand[i] for i in range(self.I))
        m.setParam('OutputFlag', 0)
        m.optimize()
        return

# two ways to add constraints
# 1. keep the basic m model.

    def solve_IP(self, m):
        xvalue = m.getVars()[0: self.I * self.given_lines]
        for var in xvalue:
            var.vtype = GRB.INTEGER
        # m.update()
        m.setParam('OutputFlag', 0)
        m.optimize()

        return m.objVal, [m.getVarByName('varx[' + str(i) + ',' + str(j) + ']').x for i in range(self.I) for j in range(self.given_lines)]

    def solveBenders(self, eps = 1e-4, maxit = 10):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype=GRB.CONTINUOUS, name='varx')
        z = m.addVars(self.W, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='varz')
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                 for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        m.addConstrs(z[i] <= 0 for i in range(self.W))
        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) + grb.quicksum(self.prop[w] * z[w] for w in range(self.W)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        var = np.array(m.getAttr('X'))
        x0 = var[0: self.I*self.given_lines]
        zstar = var[-self.W:]


        tol = float("inf")
        it = 0
        # UB = float("inf")
        LB = 0  # Any feasible solution gives a lower bound.
        while eps < tol and it < maxit:
            start1 = time.time()
            alpha_set, w_set, LB = self.add_Constrs(x0, zstar)  # give the constraints
            # m.addConstrs(grb.quicksum(alpha_set[t][i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) + z[w_set[t]-1] <= grb.quicksum(alpha_set[t][i] * self.dw[w_set[t]-1][i] for i in range(self.I)) for t in range(len(w_set)))

            # m.addConstrs(grb.quicksum(alpha_set[t][i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) + z[k-1] <= grb.quicksum(alpha_set[t][i] * self.dw[k-1][i] for i in range(self.I)) for t,k in enumerate(w_set))
            x_var = m.getVars()[0: self.I * self.given_lines]
            for t in range(len(w_set)):
                coff = alpha_set[t].repeat(self.given_lines)
                coff = coff.tolist()
                myrhs = np.dot(alpha_set[t], dw[w_set[t]-1])
                m.addLConstr(grb.LinExpr(coff, x_var), GRB.LESS_EQUAL, -z[w_set[t]-1]+myrhs)


            # print("LP and AddConstrs took...", round(time.time() - start1, 3), "seconds")
            # UB = min(UB, obj)
            m.optimize()
            obj = m.objVal
            var = np.array(m.getAttr('X'))
            x0 = var[0: self.I * self.given_lines]
            zstar = var[-self.W:]

            tol = abs(obj - LB)
            it += 1
            # print('----------------------iteration ' + str(it) + '-------------------')
            # print('LB = ', LB, ', UB = ', obj, ', tol = ', tol)
            # print('optimal solution:', newx)
        # print('The number of iterations is:', it)
        start = time.time()
        obj_IP, x0 = self.solve_IP(m)
        newx = np.reshape(x0, (self.I, self.given_lines))
        # print('each row:', newx)
        newd = np.sum(newx, axis=1)
        # print("IP took...", round(time.time() - start, 3), "seconds")
        print('optimal demand:', newd)
        # print('optimal solution:', newx)
        # print('optimal LP objective:', obj)
        # print('optimal IP objective:', obj_IP)
        return newd, LB

    def solveBendersDynamic(self, demand, eps=1e-4, maxit=10):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0,
                      vtype=GRB.CONTINUOUS, name='varx')
        z = m.addVars(self.W, lb=-float('inf'),
                      vtype=GRB.CONTINUOUS, name='varz')
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        
        # add demand constraints
        m.addConstrs(grb.quicksum(x[i,j] for j in range(self.given_lines)) >= demand[i] for i in range(self.I))

        m.addConstrs(z[i] <= 0 for i in range(self.W))
        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) + grb.quicksum(self.prop[w] * z[w] for w in range(self.W)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        var = np.array(m.getAttr('X'))
        x0 = var[0: self.I*self.given_lines]
        zstar = var[-self.W:]

        tol = float("inf")
        it = 0
        # UB = float("inf")
        LB = 0  # Any feasible solution gives a lower bound.
        while eps < tol and it < maxit:
            start1 = time.time()
            alpha_set, w_set, LB = self.add_Constrs(x0, zstar)  # give the constraints

            x_var = m.getVars()[0: self.I * self.given_lines]
            for t in range(len(w_set)):
                coff = alpha_set[t].repeat(self.given_lines)
                coff = coff.tolist()
                myrhs = np.dot(alpha_set[t], dw[w_set[t]-1])
                m.addLConstr(grb.LinExpr(coff, x_var),
                             GRB.LESS_EQUAL, -z[w_set[t]-1]+myrhs)

            print("LP and AddConstrs took...", round(time.time() - start1, 3), "seconds")
            # UB = min(UB, obj)
            m.optimize()
            obj = m.objVal
            var = np.array(m.getAttr('X'))
            x0 = var[0: self.I * self.given_lines]
            zstar = var[-self.W:]

            tol = abs(obj - LB)
            it += 1
            print('----------------------iteration ' +
                  str(it) + '-------------------')
            print('LB = ', LB, ', UB = ', obj, ', tol = ', tol)
            # print('optimal solution:', newx)
        # print('The number of iterations is:', it)
        start = time.time()
        obj_IP, x0 = self.solve_IP(m)
        newx = np.reshape(x0, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)
        print("IP took...", round(time.time() - start, 3), "seconds")
        print('optimal demand:', newd)
        # print('optimal solution:', newx)
        # print('optimal LP objective:', obj)
        print('optimal IP objective:', obj_IP)
        return newd, LB

class originalModel():
    def __init__(self, roll_width, given_lines, demand_width_array, W, I, prop, dw):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array - 1
        self.W = W
        self.I = I
        self.dw = dw
        self.prop = prop
        seat = np.insert(demand_width_array, 0, 1)
        self.seat_value = np.diff(seat)

    def Wmatrix(self):
        # n is the dimension of group types
        return - np.identity(self.I) + np.eye(self.I, k=1)

    def solveModelGurobi(self):
        m2 = grb.Model()
        x = m2.addVars(self.I, self.given_lines, lb=0,
                       vtype= GRB.INTEGER, name='varx')
        y1 = m2.addVars(self.I, self.W, lb=0,  vtype= GRB.CONTINUOUS)
        y2 = m2.addVars(self.I, self.W, lb=0,  vtype= GRB.CONTINUOUS)
        W0 = self.Wmatrix()
        start = time.time()
        m2.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                   for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        M_identity = np.identity(self.I)

        m2.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) + grb.quicksum(W0[i, j] * y1[j, w] +  M_identity[i, j]*y2[j, w] for j in range(self.I)) == self.dw[w][i] for i in range(self.I) for w in range(self.W))
        print("Constructing second took...", round(time.time() - start, 2), "seconds")
        m2.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) - grb.quicksum(self.seat_value[i]*y1[i, w]*self.prop[w] for i in range(self.I) for w in range(self.W)), GRB.MAXIMIZE)

        m2.setParam('OutputFlag', 0)
        m2.write('revisedModel.lp')
        m2.optimize()
        print('optimal value:', m2.objVal)
        sol = np.array(m2.getAttr('X'))
        solx = sol[0:self.I * self.given_lines]
        newx = np.reshape(solx, (self.I, self.given_lines))
        # print('each row:', newx)
        newd = np.sum(newx, axis=1)
        print('optimal demand:', newd)
        return m2.objVal

    def solveModelGurobiDynamic(self, demand):
        m2 = grb.Model()
        x = m2.addVars(self.I, self.given_lines, lb=0,
                       vtype=GRB.INTEGER, name='varx')
        y1 = m2.addVars(self.I, self.W, lb=0,  vtype=GRB.CONTINUOUS)
        y2 = m2.addVars(self.I, self.W, lb=0,  vtype=GRB.CONTINUOUS)
        W0 = self.Wmatrix()
        start = time.time()
        m2.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                   for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        M_identity = np.identity(self.I)

        # add demand constraints
        m2.addConstrs(grb.quicksum(x[i,j] for j in range(self.given_lines)) >= demand[i] for i in range(self.I))

        m2.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) + grb.quicksum(W0[i, j] * y1[j, w] +
                      M_identity[i, j]*y2[j, w] for j in range(self.I)) == self.dw[w][i] for i in range(self.I) for w in range(self.W))
        # print("Constructing second took...", round(time.time() - start, 2), "seconds")
        m2.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) - grb.quicksum(self.seat_value[i]*y1[i, w]*self.prop[w] for i in range(self.I) for w in range(self.W)), GRB.MAXIMIZE)

        m2.setParam('OutputFlag', 0)
        m2.optimize()
        # print('optimal value:', m2.objVal)
        sol = np.array(m2.getAttr('X'))
        solx = sol[0:self.I * self.given_lines]
        newx = np.reshape(solx, (self.I, self.given_lines))
        # print('each row:', newx)
        newd = np.sum(newx, axis=1)
        # print('optimal demand:', newd)
        return newd, m2.objVal

# print('----------------------For the whole model-------------------')
# print('The optimal value for the whole model:', t)
# print('The optimal solution for the whole model:', bb[0:I])


# when demand =0, return used demands, remaining period, decision_list
# Use remaining period, generate new dw and + used demands -> new scenarios
# call benders again.
