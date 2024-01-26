import numpy as np
from typing import List
import copy
from Method1 import stochasticModel
from SamplingMethodSto import samplingmethod1
from Method10 import deterministicModel

# Calculate the number of y_{i+1}^{+} in all scenarios
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
# Calculate y_i and give the stats.

def fractional_cal_y(self, i, d0):
# Calculate (i+1)th element for group type i
    acc = 0
    deny = 0
    for w in range(self.W):
        yplus, _ = self.obtainY(self.dw[w], d0)
        if 0 < yplus[i]:
            acc += min(yplus[i],1)
        else:
            deny += 1
    p_acc = acc/self.W
    p_deny = deny/self.W
    if i * p_acc >= (i+1) * p_deny:
        return True
    else:
        return False

def int_cal_y(self, i, d0, dw):
    # Calculate (i+1)th element for group type i
    acc = 0
    for w in range(len(dw)):
        yplus, _ = self.obtainY(dw[w], d0)
        if 0 < yplus[i]:
            acc += 1
    p_acc = acc/len(dw)
    if i * p_acc >= (i+1) * (1-p_acc):
        return True
    else:
        return False

def method_scenario(self, sequence: List[int], newx, change_roll0):
    
    change_roll = copy.deepcopy(change_roll0)
    newx = newx.T.tolist()
    mylist = []
    periods = len(sequence)

    for num, j in enumerate(sequence):
        newd = np.sum(newx, axis=0)
        remaining_period = periods - num

        if newd[j-1-self.s] > 1e-4:
            mylist.append(1)
            k = self.break_tie(newx, change_roll, j-1-self.s)
            newx[k][j-1-self.s] -= 1
            change_roll[k] -= j

            newd = np.sum(newx, axis=0)
            # if after accept j, the supply is 0, we should generate another newx.
            if j == self.s + self.I and newd[-1] == 0:
                newx = self.rearrange(change_roll, remaining_period, j)
                newd = np.sum(newx, axis=0)
        else:
            sam = samplingmethod1(self.I, self.num_sample,
                                  remaining_period-1, self.probab, self.s)
            dw, _ = sam.get_prob_ini(j)
            decision_list = int_cal_y(j-1-self.s, newd, dw)

            change_deny = copy.deepcopy(change_roll)
            if decision_list:
                for i in range(j- self.s, self.I, 1):
                    if newd[i] > 0:
                        k = self.break_tie(newx, change_roll, i)
                        newx[k][i] -= 1
                        if i - j >= 0:
                            newx[k][int(i - j)] += 1
                    break
                change_roll[k] -= j

                change_accept = copy.deepcopy(change_roll)
                sam_multi = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
                dw_acc, prop_acc = sam_multi.get_prob()
                W_acc = len(dw_acc)
                m_acc = stochasticModel(change_accept, self.given_lines, self.demand_width_array, W_acc, self.I, prop_acc, dw_acc, self.s)
                ini_demand_acc, val_acc = m_acc.solveBenders(eps=1e-4, maxit=20)
                # ini_demand_acc = np.ceil(ini_demand_acc)

                dw_deny = dw_acc
                prop_deny = prop_acc
                W_deny = len(dw_deny)
                m_deny = stochasticModel(change_deny, self.given_lines, self.demand_width_array, W_deny, self.I, prop_deny, dw_deny, self.s)
                ini_demand_deny, val_deny = m_deny.solveBenders(eps=1e-4, maxit=20)
                # ini_demand_deny = np.ceil(ini_demand_deny)

                if val_acc + (j-self.s) < val_deny:
                    mylist.append(0)
                    deterModel = deterministicModel(change_deny, self.given_lines, self.demand_width_array, self.I, self.s)
                    _, newx = deterModel.IP_formulation(np.zeros(self.I), ini_demand_deny)

                    newx = self.full_largest(newx, change_deny)
                    newx = newx.T.tolist()
                    change_roll = change_deny
                else:
                    mylist.append(1)
                    deterModel = deterministicModel(change_accept, self.given_lines, self.demand_width_array, self.I, self.s)
                    _, newx = deterModel.IP_formulation(np.zeros(self.I), ini_demand_acc)

                    newx = self.full_largest(newx, change_accept)
                    newx = newx.T.tolist()
                    change_roll = change_accept
            else:
                mylist.append(0)

    sequence = [i-self.s for i in sequence]
    final_demand = np.array(sequence) * np.array(mylist)
    final_demand = final_demand[final_demand != 0]

    demand = np.zeros(self.I)
    for i in final_demand:
        demand[i-1] += 1
    return demand


def method_scenario_1(self, sequence: List[int], newx, change_roll0):
    change_roll = copy.deepcopy(change_roll0)
    newx = newx.T.tolist()
    mylist = []
    periods = len(sequence)

    for num, j in enumerate(sequence):
        newd = np.sum(newx, axis=0)
        remaining_period = periods - num

        if newd[j-1-self.s] > 1e-4:
            mylist.append(1)
            k = self.break_tie(newx, change_roll, j-1-self.s)
            newx[k][j-1-self.s] -= 1
            change_roll[k] -= j

            newd = np.sum(newx, axis=0)
            # if after accept j, the supply is 0, we should generate another newx.
            if j == self.s + self.I and newd[-1] == 0:
                newx = self.rearrange(change_roll, remaining_period, j)
                newd = np.sum(newx, axis=0)
        else:
            sam = samplingmethod1(self.I, self.num_sample, remaining_period-1, self.probab, self.s)
            dw, _ = sam.get_prob_ini(j)
            decision_list = int_cal_y(j-1-self.s, newd, dw)

            change_deny = copy.deepcopy(change_roll)
            if decision_list:
                for i in range(j - self.s, self.I, 1):
                    if newd[i] > 0:
                        k = self.break_tie(newx, change_roll, i)
                        newx[k][i] -= 1
                        if i - j >= 0:
                            newx[k][int(i - j)] += 1
                    break
                change_roll[k] -= j

                change_accept = copy.deepcopy(change_roll)
                sam_multi = samplingmethod1(
                    self.I, self.num_sample, remaining_period-1, self.probab, self.s)
                dw_acc, prop_acc = sam_multi.get_prob()
                W_acc = len(dw_acc)
                m_acc = stochasticModel(
                    change_accept, self.given_lines, self.demand_width_array, W_acc, self.I, prop_acc, dw_acc, self.s)
                ini_demand_acc, val_acc = m_acc.solveBenders(
                    eps=1e-4, maxit=20)
                # ini_demand_acc = np.ceil(ini_demand_acc)

                dw_deny = dw_acc
                prop_deny = prop_acc
                W_deny = len(dw_deny)
                m_deny = stochasticModel(
                    change_deny, self.given_lines, self.demand_width_array, W_deny, self.I, prop_deny, dw_deny, self.s)
                ini_demand_deny, val_deny = m_deny.solveBenders(
                    eps=1e-4, maxit=20)
                # ini_demand_deny = np.ceil(ini_demand_deny)

                if val_acc + (j-self.s) < val_deny:
                    mylist.append(0)
                    deterModel = deterministicModel(
                        change_deny, self.given_lines, self.demand_width_array, self.I, self.s)
                    _, newx = deterModel.IP_formulation(
                        np.zeros(self.I), ini_demand_deny)

                    newx = self.full_largest(newx, change_deny)
                    newx = newx.T.tolist()
                    change_roll = change_deny
                else:
                    mylist.append(1)
                    deterModel = deterministicModel(
                        change_accept, self.given_lines, self.demand_width_array, self.I, self.s)
                    _, newx = deterModel.IP_formulation(
                        np.zeros(self.I), ini_demand_acc)

                    newx = self.full_largest(newx, change_accept)
                    newx = newx.T.tolist()
                    change_roll = change_accept
            else:
                mylist.append(0)

    sequence = [i-self.s for i in sequence]
    final_demand = np.array(sequence) * np.array(mylist)
    final_demand = final_demand[final_demand != 0]

    demand = np.zeros(self.I)
    for i in final_demand:
        demand[i-1] += 1
    return demand