import numpy as np
from SamplingMethod import samplingmethod
from SamplingMethodNew import samplingmethod1
from Method1 import stochasticModel
from Method10 import deterministicModel
from Mist import generate_sequence, decision1
import copy
from Mist import decisionOnce
import random

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, num_sample):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.s = 1
        self.value_array = np.arange(1, 1+I)
        self.demand_width_array = self.value_array + self.s
        self.I = I   # number, Immutable object
        self.probab = probab
        self.num_period = num_period   # number, Immutable object
        self.num_sample = num_sample   # number, Immutable object        

    def random_generate(self):
        sequence = generate_sequence(self.num_period, self.probab, self.s)
        sam = samplingmethod(self.I, self.num_sample, self.num_period-1, self.probab, sequence[0], self.s)

        dw, prop = sam.get_prob()
        W = len(dw)
        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw)

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)
        ini_demand = np.ceil(ini_demand)
        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I)

        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand)
        ini_demand, newx4 = deter.IP_formulation(ini_demand, np.zeros(self.I))
        for new_num, new_i in enumerate(newx4.T):
            occu = np.dot(new_i, self.demand_width_array)
            if occu < self.roll_width[new_num]:
                for d_num, d_i in enumerate(new_i):
                    if d_i > 0 and d_num + self.s >= self.I-1:
                        new_i[d_num] -= 1
                        new_i[self.I-1] += 1
                        break
                    elif d_i > 0:
                        new_i[d_num] -= 1
                        new_i[d_num + self.s] += 1
                        break

        ini_demand1 = np.array(self.probab) * self.num_period
        ini_demand3, _ = deter.IP_formulation(np.zeros(self.I), ini_demand1)

        ini_demand3, newx3 = deter.IP_formulation(ini_demand3, np.zeros(self.I))

        for new_num, new_i in enumerate(newx3.T):
            occu = np.dot(new_i, self.demand_width_array)
            if occu < self.roll_width[new_num]:
                for d_num, d_i in enumerate(new_i):
                    if d_i > 0 and d_num + self.s >= self.I-1:
                        new_i[d_num] -= 1
                        new_i[self.I-1] += 1
                        break
                    elif d_i > 0:
                        new_i[d_num] -= 1
                        new_i[d_num + self.s] += 1
                        break
        return sequence, ini_demand, ini_demand3, newx3, newx4

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        sequence = [i-1 for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)

        return newd

    def method_new(self, sequence, newx, change_roll0):
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)
        
        for num, j in enumerate(sequence):
            print(f'roll: {change_roll}')
            print(f'seq: {j}')
            newd = np.sum(newx, axis=0)

            remaining_period = periods - num
            if newd[j-2] > 0:
                mylist.append(1)
                for k, pattern in enumerate(newx):
                    if pattern[j-2] > 0:
                        newx[k][j-2] -= 1
                        change_roll[k] -= j
                        break

                    if k == len(newx)-1:
                        for kk, pat in enumerate(newx):
                            if pat[j-2] > 0:
                                newx[kk][j-2] -= 1
                                change_roll[kk] -= j
                                break

            else:
                usedDemand, decision_list = decisionOnce(sequence[-remaining_period:], newd, self.probab, self.s)
                Indi_Demand = np.dot(usedDemand, range(self.I))

                change_deny = copy.deepcopy(change_roll)
                if decision_list:
                    newx0 = copy.deepcopy(newx)
                    for k, pattern in enumerate(newx):
                        if pattern[decision_list] > 0:
                            newx[k][decision_list] -= 1
                            if decision_list - Indi_Demand - 2 >= 0:
                                newx[k][int(decision_list -Indi_Demand - 2)] += 1
                            change_roll[k] -= (Indi_Demand+2)
                            break
                        if k == len(newx)-1:
                            for jj, pat in enumerate(newx):
                                if pat[decision_list] > 0:
                                    newx[jj][decision_list] -= 1
                                    if decision_list - Indi_Demand - 2 >= 0:
                                        newx[jj][int(decision_list - Indi_Demand - 2)] += 1
                                    change_roll[jj] -= (Indi_Demand+2)
                                    break
                    change_accept = copy.deepcopy(change_roll)

                    sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)

                    dw_acc, prop_acc = sam_multi.accept_sample(sequence[-remaining_period:][0])

                    W_acc = len(dw_acc)
                    m_acc = stochasticModel(change_accept, self.given_lines,
                                            self.demand_width_array, W_acc, self.I, prop_acc, dw_acc)
                    ini_demand_acc, val_acc = m_acc.solveBenders(eps=1e-4, maxit=20)

                    dw_deny, prop_deny = sam_multi.get_prob()
                    W_deny = len(dw_deny)
                    m_deny = stochasticModel(change_deny, self.given_lines,
                                             self.demand_width_array, W_deny, self.I, prop_deny, dw_deny)
                    ini_demand_deny, val_deny = m_deny.solveBenders(eps=1e-4, maxit=20)
                    if val_acc + (j-1) < val_deny:
                        mylist.append(0)
                        deterModel = deterministicModel(change_deny, self.given_lines, self.demand_width_array, self.I)
                        ini_demand1 = np.ceil(ini_demand_deny)
                        ini_demand2, _ = deterModel.IP_formulation(np.zeros(self.I), ini_demand1)
                        ini_demand1, newx = deterModel.IP_formulation(ini_demand2, np.zeros(self.I))

                        for new_num, new_i in enumerate(newx.T):
                            occu = np.dot(new_i, self.demand_width_array)
                            if occu < change_deny[new_num]:
                                for d_num, d_i in enumerate(new_i):
                                    if d_i > 0 and d_num + self.s >= self.I-1:
                                        new_i[d_num] -= 1
                                        new_i[self.I-1] += 1
                                        break
                                    elif d_i > 0:
                                        new_i[d_num] -= 1
                                        new_i[d_num + self.s] += 1
                                        break
                        newx = newx.T.tolist()
                        change_roll = change_deny

                    else:
                        mylist.append(1)
                        deterModel = deterministicModel(change_accept, self.given_lines, self.demand_width_array, self.I)
                        ini_demand1 = np.ceil(ini_demand_acc)
                        ini_demand2, _ = deterModel.IP_formulation(np.zeros(self.I), ini_demand1)
                        ini_demand1, newx = deterModel.IP_formulation(ini_demand2, np.zeros(self.I))

                        for new_num, new_i in enumerate(newx.T):
                            occu = np.dot(new_i, self.demand_width_array)
                            if occu < change_accept[new_num]:
                                for d_num, d_i in enumerate(new_i):
                                    if d_i > 0 and d_num + self.s >= self.I-1:
                                        new_i[d_num] -= 1
                                        new_i[self.I-1] += 1
                                        break
                                    elif d_i > 0:
                                        new_i[d_num] -= 1
                                        new_i[d_num + self.s] += 1
                                        break

                        newx = newx.T.tolist()
                        change_roll = change_accept
                else:
                    mylist.append(0)

        sequence = [i-1 for i in sequence]

        final_demand = np.array(sequence) * np.array(mylist)
        final_demand = final_demand[final_demand != 0]
        
        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

    def method1(self, sequence, ini_demand):
        decision_list = decision1(sequence, ini_demand, self.probab, self.s)
        sequence = [i-1 for i in sequence if i > 0]

        final_demand = np.array(sequence) * np.array(decision_list)
        # print('The result of Method 1--------------')
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand


if __name__ == "__main__":
    given_lines = 10
    roll_width = np.ones(given_lines) * 21
    I = 4
    probab = np.array([0.15, 0.1, 0.6, 0.15])
    num_period = random.randint(50, 80)
    num_sample = 1000
    a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)
    sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
    g = a_instance.method_new(sequence, newx4, roll_width)
