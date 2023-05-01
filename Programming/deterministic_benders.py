import gurobipy as grb
from gurobipy import GRB
import numpy as np
import time
from collections import Counter
from scipy.stats import binom
import copy
# This function is used to calculate the revised benders (without benders decomposition)
# Use another way to update 

class deterministicModel:
    def __init__(self, roll_width, given_lines, demand_width_array, I):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = demand_width_array
        self.value_array = demand_width_array - 1
        self.I = I

    def IP_formulation(self, demand_lower, demand_upper):
        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0, vtype=GRB.INTEGER)
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                    for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        m.addConstrs(grb.quicksum(x[i, j] for j in range(self.given_lines)) <= demand_upper[i] for i in range(self.I))

        m.addConstrs(grb.quicksum(x[i,j] for j in range(self.given_lines)) >= demand_lower[i] for i in range(self.I))

        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)), GRB.MAXIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        # print('************************************************')
        # print('Optimal value of IP is: %g' % m.objVal)
        x_ij = np.array(m.getAttr('X'))
        newx = np.reshape(x_ij, (self.I, self.given_lines))
        newd = np.sum(newx, axis=1)
        return newd, m.objVal

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

    def solve_IP(self, m):
        xvalue = m.getVars()[0: self.I * self.given_lines]
        for var in xvalue:
            var.vtype = GRB.INTEGER
        m.update()
        m.setParam('OutputFlag', 1)
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
            print('----------------------iteration ' + str(it) + '-------------------')
            print('LB = ', LB, ', UB = ', obj, ', tol = ', tol)
            # print('optimal solution:', newx)
        # print('The number of iterations is:', it)
        start = time.time()
        # obj_IP, x0 = self.solve_IP(m)
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

class originalModel:
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
        print('optimal demand:', newd)
        return newd, m2.objVal

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
        m2.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) - grb.quicksum(
            self.seat_value[i]*y1[i, w]*self.prop[w] for i in range(self.I) for w in range(self.W)), GRB.MAXIMIZE)
        start = time.time()
        m2.setParam('OutputFlag', 0)
        m2.optimize()
        # print('optimal value:', m2.objVal)
        sol = np.array(m2.getAttr('X'))
        solx = sol[0:self.I * self.given_lines]
        newx = np.reshape(solx, (self.I, self.given_lines))
        # print('each row:', newx)
        newd = np.sum(newx, axis=1)
        # print('optimal demand:', newd)
        print('It took', round(time.time()-start, 2), 'seconds')
        return newd, m2.objVal

class samplingmethod:
    def __init__(self, I, number_sample, number_period, prob) -> None:
        self.I = I
        self.number_sample = number_sample
        self.number_period = number_period
        # sample_multi = np.random.multinomial(self.number_period, [1/self.I]*self.I, size=self.number_sample)
        sample_multi = np.random.multinomial(self.number_period, prob, size=self.number_sample)
        self.sample_multi = sample_multi.tolist()

    def convert(self):
        # Converting integer list to string list
        number_arrivals = 0
        for i in self.sample_multi:
            s = [str(j) for j in i]
        # Join list items using join()
            res = " ".join(s)
            self.sample_multi[number_arrivals] = res
            number_arrivals += 1
        return self.sample_multi

    # counter_list = convert()
    # sample_result = Counter(counter_list)

    def get_prob(self):
        # Return the scenarios and the corresponding probability through sampling.
        counter_list = self.convert()
        sample_result = Counter(counter_list)
        number_scenario = len(sample_result)
        scenario_set = np.array([[0]*self.I] * number_scenario)
        count = 0
        prob_set = [0] * number_scenario
        for key, value in sample_result.items():
            key = key.split()
            int1 = [int(i) for i in key]
            prob = value / self.number_sample
            scenario_set[count] = int1
            prob_set[count] = prob
            count += 1

        return scenario_set, np.array(prob_set)

def several_class(size_group, demand, remaining_period, probab):
    # The function is used to give the maximum difference and give the decision
    # j F(x_j -1, T, p_j) - (j-i-1) F(x_{j-i-1}, T, p_{j-i-1}) -1
    # here probab should be a vector of arrival rate
    # demand is the current left demand
    # size_group is the actual size of group i
    max_size = I
    if size_group == max_size:
        return False
    diff_set = np.zeros(max_size - size_group)
    count = 0
    for j in range(size_group+1, max_size+1, 1):
        term1 = j * binom.cdf(demand[j-1]-1, remaining_period, probab[j-1])
        term2 = (j-size_group-1) * binom.cdf(demand[j-size_group-2],remaining_period, probab[j-size_group-2])
        diff_set[count] = term1 - term2 - 1
        count += 1
    max_diff = max(diff_set)
    index_diff = np.argmax(diff_set) + size_group
    if max_diff > 0:
        return index_diff
    else:
        return False

def decision1(sequence, demand, probab):
    # the function is used to make a decision on several classes
    # sequence is one possible sequence of the group arrival.
    period = len(sequence)
    group_type = sorted(list(set(sequence)))
    decision_list = [0] * period
    t = 0
    for i in sequence:
        remaining_period = period - t
        position = group_type.index(i)
        demand_posi = demand[position]
        if demand_posi > 0:
            decision_list[t] = 1
            demand[position] = demand_posi - 1
        elif sum(demand) == 0:
            break
        elif i == group_type[-1] and demand[-1]==0:
            decision_list[t] = 0
        else:
            accept_reject = several_class(i-1, demand, remaining_period-1, probab)
            if accept_reject:
                decision_list[t] = 1
                demand[accept_reject] -= 1
                if accept_reject-position-2 >= 0:
                    demand[accept_reject-position-2] += 1
        t += 1
        # print('the period:', t)
        # print('demand is:', demand)
    return decision_list

def generate_sequence(period, prob):
    trials = [np.random.choice([2, 3, 4, 5], p=prob) for i in range(period)]
    return trials


def decision_demand(sequence, decision_list):
    accept_list = np.multiply(sequence, decision_list)
    dic = Counter(accept_list)
    # Sort the list according to the value of dictionary.
    res_demand = [dic[key] for key in sorted(dic)]
    return res_demand

# print('----------------------For the whole model-------------------')
# print('The optimal value for the whole model:', t)
# print('The optimal solution for the whole model:', bb[0:I])

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    number_period = 80
    given_lines = 8
    np.random.seed(0)
    # dw = np.random.randint(20, size=(W, I)) + 20
    # dw = np.random.randint(low = 50, high= 100, size=(W, I))
    probab = [0.4, 0.4, 0.1, 0.1]
    sam = samplingmethod(I, num_sample, number_period, probab)

    dw, prop = sam.get_prob()
    W = len(dw)

    # roll_width = np.random.randint(21, size = given_lines) + 30
    roll_width = np.arange(21, 21 + given_lines)
    # total_seat = np.sum(roll_width)

    demand_width_array = np.arange(2, 2+I)

    sequence = generate_sequence(number_period, probab)
    sequence1 = copy.deepcopy(sequence)

    my = stochasticModel(roll_width, given_lines,demand_width_array, W, I, prop, dw)

    # my1 = originalModel(roll_width, given_lines, demand_width_array, W, I, prop, dw)

    # ini_demand, upperbound = my1.solveModelGurobi()

    ini_demand, upperbound = my.solveBenders(eps = 1e-4, maxit= 20)

    decision_list = decision1(sequence, ini_demand, probab)
    total_people = np.dot(sequence, decision_list)
    final_demand = np.array(sequence) * np.array(decision_list)
    # print(final_demand)
    print(total_people)
    print(Counter(final_demand))
    
# when demand =0, return used demands, remaining period, decision_list
# Use remaining period, generate new dw and + used demands -> new scenarios
# call benders again.


def decisionOnce(sequence, demand, probab):
    # the function is used to make a decision once on several classes
    # sequence is one possible sequence of the group arrival.
    record_demand = np.zeros(len(demand))
    period = len(sequence)
    group_type = sorted(list(set(sequence)))
    decision_list = 0
    i = sequence[0]
    remaining_period = period
    position = group_type.index(i)

    if i == group_type[-1] and demand[-1] == 0:
        decision_list = 0
    else:
        accept_reject = several_class(
            i-1, demand, remaining_period-1, probab)
        if accept_reject:
            decision_list = accept_reject
            demand[accept_reject] -= 1
            if accept_reject-position-2 >= 0:
                demand[accept_reject-position-2] += 1
            record_demand[position] = 1
    return record_demand, decision_list

def decisionSeveral(sequence, demand):
    # the function is used to make several decisions
    # Sequence is one possible sequence of the group arrival.
    period = len(sequence)
    group_type = sorted(list(set(sequence)))
    # decision_list = [0] * period
    originalDemand = copy.deepcopy(demand)
    t = 0
    for i in sequence:
        remaining_period = period - t
        position = group_type.index(i)
        demand_posi = demand[position]
        if demand_posi > 0:
            # decision_list[t] = 1
            demand[position] = demand_posi - 1
        else:
            usedDemand = originalDemand - demand
            break
        t += 1
        # print('the period:', t)
        # print('demand is:', demand)
    # decision_list = decision_list[0:t]
    return usedDemand, remaining_period

def newScenario(usedDemand, remaining_period):

    sam1 = samplingmethod(I, num_sample, remaining_period, probab)

    dw, prop = sam1.get_prob()
    newDemand = dw + usedDemand
    return newDemand, prop


total_usedDemand = np.zeros(I)
ini_demand1 = np.array(probab) * number_period

deterModel = deterministicModel(roll_width, given_lines, demand_width_array, I)

ini_demand, obj = deterModel.IP_formulation(total_usedDemand, ini_demand1)

mylist = []
remaining_period0 = number_period

for i in range(150):
    demand = ini_demand - total_usedDemand

    usedDemand, remaining_period = decisionSeveral(sequence, demand)
    
    diff_period = remaining_period0 - remaining_period

    mylist += [1] * diff_period

    if any(usedDemand) == 0:  # all are 0
        usedDemand, decision_list = decisionOnce(sequence, demand, probab)
        if decision_list:
            mylist.append(1)
        else:
            mylist.append(0)
        remaining_period -= 1

    remaining_period0 = remaining_period
    sequence = sequence[-remaining_period:]

    total_usedDemand += usedDemand
    
    ini_demand1 = total_usedDemand + np.ceil(np.array(probab) * remaining_period)

    deterModel = deterministicModel(roll_width, given_lines, demand_width_array, I)

    ini_demand, obj = deterModel.IP_formulation(total_usedDemand, ini_demand1)
    if len(sequence) < 10:
        break


remaining_demand = ini_demand - total_usedDemand
decision_list = decision1(sequence, remaining_demand, probab)


mylist += decision_list

total_people1 = np.dot(sequence1, mylist)
final_demand1 = np.array(sequence1) * np.array(mylist)
# print(final_demand1)
print(total_people1)
print(Counter(final_demand1))
print(Counter(sequence1))
