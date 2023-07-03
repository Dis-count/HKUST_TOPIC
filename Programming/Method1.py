import gurobipy as grb
from gurobipy import GRB
import numpy as np
import time
from SamplingMethod import samplingmethod
from Mist import generate_sequence, decision1
from collections import Counter
from Method3 import deterministicModel

# This function uses benders' decomposition to solve stochastic Model directly.
# And give the once decision.

class stochasticModel:
    def __init__(self, roll_width, given_lines, demand_width_array, W, I, prop, dw):
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

    def value(self, d0):
        part1 = self.value_array @ d0
        part2 = 0
        for w in range(self.W):
            yplus, _ = self.obtainY(self.dw[w], d0)
            part2 -= self.prop[w] * sum(yplus)

        obj_value = part1 + part2

        return obj_value

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

    def add_Constrs(self, xstar, zstar):
        # xstar is the [I* given_lines, 1] vector
        newx = np.reshape(xstar, (self.I, self.given_lines))
        d0 = np.sum(newx, axis=1)
        LB = self.value_array @ d0
        alphaset = []
        wset = []

        for w in range(self.W):
            yplus, yminus = self.obtainY(self.dw[w], d0)
            alpha = self.solveSub(yplus, yminus)  # alpha is a list
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
        if len(wset) == 0:
            return m, m.objVal, m.getAttr(GRB.Attr.X, m.getVars()[0: self.I * self.given_lines]), m.getAttr(GRB.Attr.X, m.getVars()[-self.W:])
        # print('wset:', len(wset))
        x_var = m.getVars()[0: self.I * self.given_lines]
        z_var = m.getVars()[-self.W:]

        for t, k in enumerate(wset):
            # m.addConstr(grb.quicksum(alphaset[t][i] * m.getVarByName('varx[' + str(i) + ',' + str(j) + ']') for i in range(self.I) for j in range(self.given_lines)) + m.getVarByName('varz[' + str(k-1) + ']') <= grb.quicksum(alphaset[t][i] * self.dw[k-1][i] for i in range(self.I)))
            m.addConstr(grb.quicksum(alphaset[t][i] * x_var[i*self.given_lines+j] for i in range(self.I) for j in range(
                self.given_lines)) + z_var[k-1] <= grb.quicksum(alphaset[t][i] * self.dw[k-1][i] for i in range(self.I)))
        # [m.getVarByName('varx[' + str(i) + ',' + str(j) + ']').x for i in range(self.I) for j in range(self.given_lines)]
        # [m.getVarByName('varz[' + str(w) + ']').x for w in range(self.W)]
        print("Constructing LP took...", round(
            time.time() - start, 3), "seconds")
        m.setParam('OutputFlag', 0)
        # m.write('Master1.lp')
        m.optimize()

        return m, m.objVal, m.getAttr(GRB.Attr.X, m.getVars()[0: self.I * self.given_lines]), m.getAttr(GRB.Attr.X, m.getVars()[-self.W:])

    def addDemand(self, m, demand):
        # This function is used to add demand constraints to the model.
        x_var = m.getVars()[0: self.I * self.given_lines]
        m.addConstrs(grb.quicksum(x_var[i*self.given_lines+j]
                     for j in range(self.given_lines)) >= demand[i] for i in range(self.I))
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

    def solveBenders(self, eps=1e-4, maxit=20):

        m = grb.Model()
        x = m.addVars(self.I, self.given_lines, lb=0,
                      vtype=GRB.CONTINUOUS, name='varx')
        z = m.addVars(self.W, lb=-float('inf'),
                      vtype=GRB.CONTINUOUS, name='varz')
        m.addConstrs(grb.quicksum(self.demand_width_array[i] * x[i, j]
                                  for i in range(self.I)) <= self.roll_width[j] for j in range(self.given_lines))
        m.addConstrs(z[i] <= 0 for i in range(self.W))
        m.setObjective(grb.quicksum(self.value_array[i] * x[i, j] for i in range(self.I) for j in range(
            self.given_lines)) + grb.quicksum(self.prop[w] * z[w] for w in range(self.W)), GRB.MAXIMIZE)
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
            alpha_set, w_set, LB = self.add_Constrs(
                x0, zstar)  # give the constraints
            # m.addConstrs(grb.quicksum(alpha_set[t][i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) + z[w_set[t]-1] <= grb.quicksum(alpha_set[t][i] * self.dw[w_set[t]-1][i] for i in range(self.I)) for t in range(len(w_set)))

            # m.addConstrs(grb.quicksum(alpha_set[t][i] * x[i, j] for i in range(self.I) for j in range(self.given_lines)) + z[k-1] <= grb.quicksum(alpha_set[t][i] * self.dw[k-1][i] for i in range(self.I)) for t,k in enumerate(w_set))
            x_var = m.getVars()[0: self.I * self.given_lines]
            for t in range(len(w_set)):
                coff = alpha_set[t].repeat(self.given_lines)
                coff = coff.tolist()
                myrhs = np.dot(alpha_set[t], self.dw[w_set[t]-1])
                m.addLConstr(grb.LinExpr(coff, x_var),
                             GRB.LESS_EQUAL, -z[w_set[t]-1]+myrhs)

            # print("LP and AddConstrs took...", round(time.time() - start1, 3), "seconds")
            # UB = min(UB, obj)
            m.optimize()
            obj = m.objVal
            var = np.array(m.getAttr('X'))
            x0 = var[0: self.I * self.given_lines]
            zstar = var[-self.W:]

            tol = abs(obj - LB)
            it += 1
            # print('----------------------iteration ' +
            #       str(it) + '-------------------')
            # print('LB = ', LB, ', UB = ', obj, ', tol = ', tol)
            # print('optimal solution:', newx)
        # print('The number of iterations is:', it)
        # obj_IP, x0 = self.solve_IP(m)
        newx = np.reshape(x0, (self.I, self.given_lines))
        # print('each row:', newx)
        newd = np.sum(newx, axis=1)
        # print("IP took...", round(time.time() - start, 3), "seconds")
        # print('optimal demand:', newd)
        # print('optimal solution:', newx)
        # print('optimal LP objective:', obj)
        # print('optimal IP objective:', obj_IP)
        return newd, LB


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    number_period = 55
    given_lines = 8
    np.random.seed(0)

    probab = [0.4, 0.4, 0.1, 0.1]
    sam = samplingmethod(I, num_sample, number_period, probab)

    dw, prop = sam.get_prob()
    W = len(dw)

    roll_width = np.arange(21, 21 + given_lines)
    total_seat = np.sum(roll_width)

    demand_width_array = np.arange(2, 2+I)

    sequence = generate_sequence(number_period, probab)

    my = stochasticModel(roll_width, given_lines, demand_width_array, W, I, prop, dw)

    start = time.time()
    ini_demand, upperbound = my.solveBenders(eps=1e-4, maxit=20)
    print("Benders took...", round(time.time() - start, 3), "seconds")

    ini_demand = np.ceil(ini_demand)  # take the ceiling of the upper demand

    deter = deterministicModel(roll_width, given_lines, demand_width_array, I)

    ini_demand, obj = deter.IP_formulation(np.zeros(I), ini_demand)

    decision_list = decision1(sequence, ini_demand, probab)

    sequence = [i-1 for i in sequence if i > 0]
    total_people = np.dot(sequence, decision_list)
    final_demand = np.array(sequence) * np.array(decision_list)
    print(f'The total seats: {total_seat}')
    print(f'The total people:{total_people}')
    print(Counter(final_demand))
