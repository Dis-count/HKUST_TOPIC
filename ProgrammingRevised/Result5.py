# Impact of social distance under different demands
import numpy as np
from SamplingMethodNew import samplingmethod1
from Method1 import stochasticModel
from Method4 import deterministicModel
from Method5 import deterministicModel1
from Mist import generate_sequence, decision1
import copy
from Mist import decisionSeveral, decisionOnce
import time
import matplotlib.pyplot as plt

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, num_sample):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.demand_width_array = np.arange(2, 2+I)
        self.value_array = self.demand_width_array - 1
        self.I = I   # number, Immutable object
        self.probab = probab
        self.num_period = num_period   # number, Immutable object
        self.num_sample = num_sample   # number, Immutable object

    def random_generate(self):
        sam = samplingmethod1(self.I, self.num_sample,
                             self.num_period, self.probab)

        dw, prop = sam.get_prob()
        W = len(dw)

        sequence = generate_sequence(self.num_period, self.probab)

        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw)

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)

        deter = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)

        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand)
        ini_demand, _ = deter.IP_formulation(ini_demand, np.zeros(self.I))

        ini_demand1 = np.array(self.probab) * self.num_period

        ini_demand3, _ = deter.IP_formulation(np.zeros(self.I), ini_demand1)
        ini_demand3, _ = deter.IP_formulation(ini_demand3, np.zeros(self.I))

        return sequence, ini_demand, ini_demand3

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

    def offline1(self, sequence):
        # Optimal decision Without social distance.
        # The length is different.
        demand = np.zeros(self.I)
        sequence = [i-1 for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel1(
            self.roll_width-1, self.given_lines, self.demand_width_array-1, self.I)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)
        
        return newd

    def method_new(self, sequence, newx, change_roll0):
        # Use supply control and stochastic model value to make the decision
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        periods = len(sequence)

        for num, j in enumerate(sequence):
            newd = np.sum(newx, axis=0)

            remaining_period = periods - num
            if newd[j-2] > 0:
                mylist.append(1)
                for k, pattern in enumerate(newx):
                    if pattern[j-2] > 0 and (change_roll[k] > (self.I + 1) or change_roll[k] == j):
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
                usedDemand, decision_list = decisionOnce(
                    sequence[-remaining_period:], newd, self.probab)
                Indi_Demand = np.dot(usedDemand, range(self.I))

                change_deny = copy.deepcopy(change_roll)
                if decision_list:
                    newx0 = copy.deepcopy(newx)
                    for k, pattern in enumerate(newx):
                        if pattern[decision_list] > 0 and change_roll[k] > (self.I + 1):
                            newx[k][decision_list] -= 1
                            if decision_list - Indi_Demand - 2 >= 0:
                                newx[k][int(decision_list -
                                            Indi_Demand - 2)] += 1
                            change_roll[k] -= (Indi_Demand+2)
                            break
                        if k == len(newx)-1:
                            for jj, pat in enumerate(newx):
                                if pat[decision_list] > 0:
                                    newx[jj][decision_list] -= 1
                                    if decision_list - Indi_Demand - 2 >= 0:
                                        newx[jj][int(
                                            decision_list - Indi_Demand - 2)] += 1
                                    change_roll[jj] -= (Indi_Demand+2)
                                    break
                    change_accept = copy.deepcopy(change_roll)
                    # sam_accept = samplingmethod(I, num_sample, remaining_period-1, probab, sequence[-remaining_period:][0])
                    # dw_acc, prop_acc = sam_accept.get_prob()
                    sam_multi = samplingmethod1(
                        I, num_sample, remaining_period-1, probab)

                    dw_acc, prop_acc = sam_multi.accept_sample(
                        sequence[-remaining_period:][0])

                    W_acc = len(dw_acc)
                    m_acc = stochasticModel(change_accept, self.given_lines,
                                            self.demand_width_array, W_acc, self.I, prop_acc, dw_acc)
                    ini_demand_acc, val_acc = m_acc.solveBenders(
                        eps=1e-4, maxit=20)

                    dw_deny, prop_deny = sam_multi.get_prob()
                    W_deny = len(dw_deny)
                    m_deny = stochasticModel(change_deny, self.given_lines,
                                             self.demand_width_array, W_deny, self.I, prop_deny, dw_deny)
                    ini_demand_deny, val_deny = m_deny.solveBenders(
                        eps=1e-4, maxit=20)

                    if val_acc + (j-1) < val_deny:
                        mylist.append(0)
                        deterModel = deterministicModel(
                            change_deny, self.given_lines, self.demand_width_array, self.I)
                        ini_demand1 = np.ceil(ini_demand_deny)
                        ini_demand2, _ = deterModel.IP_formulation(
                            np.zeros(self.I), ini_demand1)
                        ini_demand1, newx = deterModel.IP_formulation(
                            ini_demand2, np.zeros(self.I))

                        for new_num, new_i in enumerate(newx.T):
                            occu = np.dot(new_i, np.arange(2, I+2))
                            if occu < change_deny[new_num]:
                                for d_num, d_i in enumerate(new_i):
                                    if d_i > 0 and d_num < I-1:
                                        new_i[d_num] -= 1
                                        new_i[d_num+1] += 1
                                        break
                        newx = newx.T.tolist()
                        change_roll = change_deny

                        # newx = newx0
                    else:
                        mylist.append(1)
                        deterModel = deterministicModel(
                            change_accept, self.given_lines, self.demand_width_array, self.I)
                        ini_demand1 = np.ceil(ini_demand_acc)
                        ini_demand2, _ = deterModel.IP_formulation(
                            np.zeros(self.I), ini_demand1)
                        ini_demand1, newx = deterModel.IP_formulation(
                            ini_demand2, np.zeros(self.I))

                        # print(f'test: {acc1}')
                        # print(f'deny_value: {val_deny}')
                        for new_num, new_i in enumerate(newx.T):
                            occu = np.dot(new_i, np.arange(2, I+2))
                            if occu < change_accept[new_num]:
                                for d_num, d_i in enumerate(new_i):
                                    if d_i > 0 and d_num < I-1:
                                        new_i[d_num] -= 1
                                        new_i[d_num+1] += 1
                                        break

                        newx = newx.T.tolist()
                        change_roll = change_accept
                else:
                    mylist.append(0)

        sequence = [i-1 for i in sequence if i > 0]

        final_demand = np.array(sequence) * np.array(mylist)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

    def method1(self, sequence, ini_demand):
        decision_list = decision1(sequence, ini_demand, self.probab)
        sequence = [i-1 for i in sequence if i > 0]

        final_demand = np.array(sequence) * np.array(decision_list)
        # print('The result of Method 1--------------')
        final_demand = final_demand[final_demand!=0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand

    def result(self, sequence, ini_demand, ini_demand3):
        ini_demand4 = copy.deepcopy(ini_demand)

        final_demand1 = self.method1(sequence, ini_demand)

        final_demand3 = self.method4(sequence, ini_demand3)

        final_demand4 = self.method4(sequence, ini_demand4)

        return final_demand1, final_demand3, final_demand4

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(10,100,1)
    given_lines = 10
    # np.random.seed(i)
    probab = [0.3, 0.2, 0.2, 0.3]

    t_value = np.arange(10, 100, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))

    cnt = 0
    gap_if = True
    for num_period in period_range:
        roll_width = np.ones(given_lines) * 21
        total_seat = np.sum(roll_width) - given_lines

        a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)
        sto = 0
        accept_people = 0

        multi = np.arange(1, I+1)
        print(num_period)
        count = 50
        for j in range(count):
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()

            f = a_instance.offline1(sequence)  # optimal result
            optimal = np.dot(multi, f)

            # d = a_instance.offline(sequence)
            g = a_instance.method_new(sequence, ini_demand, newx4, roll_width)
            sto += np.dot(multi, g)
            accept_people += optimal

        occup_value[cnt] = sto/count/total_seat * 100
        people_value[cnt] = accept_people/count/total_seat * 100
        if gap_if:
            if accept_people/count - sto/count > 1:
                point = [num_period-1, occup_value[cnt-1]]
                gap_if = False
        cnt += 1
    plt.plot(t_value, people_value, 'b-', label = 'Without social distancing')
    plt.plot(t_value, occup_value, 'r--', label = 'With social distancing')
    plt.xlabel('Periods')
    plt.ylabel('Percentage of total seats')
    point[1] = round(point[1], 2)
    plt.annotate(r'Gap $%s$' % str(point), xy=point, xytext=(
        point[0]+10, point[1]-20), arrowprops=dict(facecolor='black', shrink=0.1),)
    plt.legend()
    plt.show()
    print(point)
