import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method10 import deterministicModel
from Mist import generate_sequence, decision1
import copy
from Mist import decisionSeveral, decisionOnce
import time
# This function call different methods

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
        sam = samplingmethod(self.I, self.num_sample, self.num_period, self.probab)

        dw, prop = sam.get_prob()
        W = len(dw)

        sequence = generate_sequence(self.num_period, self.probab)

        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw)

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)

        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I)

        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand)
        ini_demand, newx4 = deter.IP_formulation(ini_demand, np.zeros(self.I))

        ini_demand1 = np.array(self.probab) * self.num_period
        ini_demand3, _ = deter.IP_formulation(np.zeros(self.I), ini_demand1)

        ini_demand3, newx3 = deter.IP_formulation(ini_demand3, np.zeros(self.I))

        return sequence, ini_demand, ini_demand3, newx3, newx4

    def row_by_row(self, sequence):
        # i is the i-th request in the sequence
        # j is the j-th row
        # sequence includes social distance.
        current_capacity = copy.deepcopy(self.roll_width)
        j = 0
        period = 0
        for i in sequence:
            if i in current_capacity:
                inx = np.where(current_capacity == i)[0][0]
                current_capacity[inx] = 0
                period += 1
                continue

            if current_capacity[j] >= i:
                current_capacity[j] -= i
            else:
                j += 1
                if j > self.given_lines-1:
                    break
                current_capacity[j] -= i
            period += 1

        lis = [0] * (self.num_period - period)
        for k, i in enumerate(sequence[period:]):
            if i in current_capacity:
                inx = np.where(current_capacity == i)[0][0]
                current_capacity[inx] = 0

                lis[k] = 1

        my_list111 = [1] * period + lis
        sequence = [i-1 for i in sequence]

        final_demand = np.array(sequence) * np.array(my_list111)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand

    def binary_search_first(self, sequence):
        # Return the index not less than the first
        target = sum(self.roll_width)
        arr = np.cumsum(sequence)
        low = 0
        high = len(arr)-1
        res = -1
        while low <= high:
            mid = (low + high)//2
            if target <= arr[mid]:
                res = mid
                high = mid-1
            else:
                low = mid+1
        if res == -1:
            total = sum(sequence)
            seq = sequence
        else:
            seq = sequence[0:res]
            total = sum(seq)

        remaining = target - total
        if remaining > 0 and res > 0:
            for i in sequence[res:]:
                if i <= remaining:
                    seq = sequence[0:res] + [i]
                    remaining -= i

        seq = [i-1 for i in seq]
        demand = np.zeros(self.I)
        for i in seq:
            demand[i-1] += 1

        deter1 = deterministicModel(self.roll_width, self.given_lines,
                                            self.demand_width_array, self.I)
        indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        while not indi:  # If indi doesnot exist, then delete the arrival from the seq. 
            demand[seq[-1]-1] -= 1
            seq.pop()
            indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        seq = [i+1 for i in seq]

        return seq

    def dynamic_program(self, sequence):
        S = int(sum(self.roll_width))
        p = self.probab
        T = self.num_period
        capa =0 # used to indicate whether the capacity is enough
        option = self.I
        value = [[0 for _ in range(T + 1)] for _ in range(S + 1)]
        record = [[[0] * option for _ in range(T + 1)] for _ in range(S+1)]
        for i in range(1, S + 1):
            for j in range(1, T + 1):
                value[i][j] = value[i][j-1]

                everyvalue = 0
                totalvalue = 0
                for k in range(option):
                    if k == (option -1) and (i - self.value_array[k]) >= 1:
                        everyvalue = value[i - self.value_array[k] -1][j - 1] + self.value_array[k]
                        capa = 1
                    elif (i - self.value_array[k]) >= 1:
                        everyvalue = value[i - self.value_array[k]-1][j - 1] + self.value_array[k]
                        capa = 1
                    else:
                        everyvalue = value[i][j-1]
                        capa = 0

                    if value[i][j-1] <= everyvalue and capa:  # delta_k
                        totalvalue += p[k] * everyvalue
                        record[i][j][k] = 1
                    else:
                        totalvalue += p[k] * value[i][j-1]
                value[i][j] = totalvalue

        decision_list = [0] * T
        sequence = [i-1 for i in sequence if i > 0]

        for k, i in enumerate(sequence):  # i = 1,2,3,4
            decision = record[S][T][i-1]
            if decision:
                S -= i+1
            T -= 1
            decision_list[k] = decision
        final_demand = np.array(sequence) * np.array(decision_list)

        final_demand = final_demand[final_demand!=0]
        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        deter1 = deterministicModel(self.roll_width, self.given_lines,
                                    self.demand_width_array, self.I)
        indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        while not indi:
            demand[final_demand[-1]-1] -= 1
            final_demand = final_demand[:-1]
            indi = deter1.IP_formulation1(demand, np.zeros(self.I))
        return demand

    def bid_price(self, sequence):
        decision_list = [0] * self.num_period
        roll_width = copy.deepcopy(self.roll_width)

        for t in range(self.num_period):
            i = sequence[t]
            if max(roll_width) < i:
                decision_list[t] = 0
            else:
                demand = (self.num_period - t) * np.array(self.probab)

                deterModel = deterministicModel(
                    roll_width, self.given_lines, self.demand_width_array, self.I)
                value, obj = deterModel.LP_formulation(demand, roll_width)
                decision = (i-1) - value * i
                for j in range(self.given_lines):
                    if roll_width[j] < i:
                        decision[j] = -1

                val = max(decision)
                decision_ind = np.array(decision).argmax()
                if val >= 0 and roll_width[decision_ind]-i >= 0:
                    decision_list[t] = 1
                    roll_width[decision_ind] -= i
                else:
                    decision_list[t] = 0

        sequence = [i-1 for i in sequence]
        final_demand = np.array(sequence) * np.array(decision_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(I)
        for i in final_demand:
            demand[i-1] += 1
        return demand

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

    def loss(self, demand):
        multi = np.arange(1, I+1)
        people = np.dot(multi, demand)
        total_seat = np.sum(roll_width)
        loss1 = total_seat - people
        return loss1

    def method4(self, sequence, ini_demand, newx, change_roll0):
        # newx patterns for N rows.
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        remaining_period0 = self.num_period
        sequence1 = copy.copy(sequence)
        total_usedDemand = np.zeros(self.I)
        # ini_demand1 = np.array(self.probab) * self.num_period
        deterModel = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)
        
        while remaining_period0:
            demand = ini_demand
            
            usedDemand, remaining_period = decisionSeveral(sequence, demand)

            diff_period = remaining_period0 - remaining_period

            demand_list = sequence[0:diff_period]

            for j in demand_list:
                for k, pattern in enumerate(newx):
                    if pattern[j-2] > 0 and (change_roll[k] > (self.I + 1) or change_roll[k] == j):
                        newx[k][j-2] -= 1
                        change_roll[k] -= j
                        break

                    if k == len(newx)-1:
                        # for t, i in enumerate(change_roll):
                        #     newx[t][i-2] = 1
                        for kk, pat in enumerate(newx):
                            if pat[j-2] > 0:
                                newx[kk][j-2] -= 1
                                change_roll[kk] -= j
                                break


            mylist += [1] * diff_period

            if any(usedDemand) == 0:  # all are 0
                usedDemand, decision_list = decisionOnce(
                    sequence, demand, self.probab)
                # print(f'Decision: {decision_list}')
                Indi_Demand = np.dot(usedDemand, range(self.I))

                if decision_list:
                    mylist.append(1)

                    # find the row can assign usedDemand（j)
                    for k, pattern in enumerate(newx):
                        if pattern[decision_list] > 0 and change_roll[k] > (self.I +1):
                            newx[k][decision_list] -= 1
                            if decision_list - Indi_Demand - 2 >= 0:
                                newx[k][int(decision_list - Indi_Demand - 2)] += 1
                            change_roll[k] -= (Indi_Demand+2)
                            break
                        if k == len(newx)-1:
                            for j, pat in enumerate(newx):
                                if pat[decision_list] > 0:
                                    newx[j][decision_list] -= 1
                                    if decision_list - Indi_Demand - 2 >= 0:
                                        newx[j][int(decision_list - Indi_Demand - 2)] += 1
                                    change_roll[j] -= (Indi_Demand+2)
                                    break

                else:
                    mylist.append(0)
                remaining_period -= 1

            remaining_period0 = remaining_period
            sequence = sequence[-remaining_period:]

            total_usedDemand += usedDemand
            
            # sam = samplingmethod(I, num_sample, remaining_period0, probab)
            # dw, prop = sam.get_prob()
            # W = len(dw)
            # m1 = stochasticModel(change_roll, self.given_lines,
            #                      self.demand_width_array, W, self.I, prop, dw)

            # ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)


            # #  use stochastic calculate
            # ini_demand, _ = deterModel.IP_formulation(
            #     np.zeros(self.I), ini_demand)
            # ini_demand, newx = deterModel.IP_formulation(ini_demand, np.zeros(self.I))
            
            # use deterministic calculate
            ini_demand, newx = deterModel.IP_formulation2(change_roll, remaining_period0, self.probab)
            newx = newx.T.tolist()

        sequence1 = [i-1 for i in sequence1 if i > 0]

        final_demand1 = np.array(sequence1) * np.array(mylist)
        final_demand1 = final_demand1[final_demand1 != 0]

        demand = np.zeros(self.I)
        for i in final_demand1:
            demand[i-1] += 1
        return demand

    def method6(self, sequence, ini_demand, newx, change_roll0):
        # loss
        # newx patterns for N rows.
        total_loss = self.loss(ini_demand)
        current_loss = 0
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        remaining_period0 = copy.deepcopy(self.num_period)
        sequence1 = copy.copy(sequence)
        total_usedDemand = np.zeros(self.I)
        # ini_demand1 = np.array(self.probab) * self.num_period
        deterModel = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)

        while remaining_period0:
            demand = ini_demand

            usedDemand, remaining_period = decisionSeveral(sequence, demand)

            diff_period = remaining_period0 - remaining_period

            demand_list = sequence[0:diff_period]

            for j in demand_list:
                for k, pattern in enumerate(newx):
                    if pattern[j-2] > 0 and (change_roll[k] > (self.I + 1) or change_roll[k] == j):
                        newx[k][j-2] -= 1
                        change_roll[k] -= j
                        break

                    if k == len(newx)-1:
                        # for t, i in enumerate(change_roll):
                        #     newx[t][i-2] = 1
                        for kk, pat in enumerate(newx):
                            if pat[j-2] > 0:
                                newx[kk][j-2] -= 1
                                change_roll[kk] -= j
                                break

            mylist += [1] * diff_period
            current_loss += diff_period
            
            if  remaining_period0 > 6:
                if any(usedDemand) == 0 and (current_loss+2)/(self.num_period - remaining_period + 1) < total_loss/self.num_period:  # all are 0

                    usedDemand, decision_list = decisionOnce(
                        sequence, demand, self.probab)
                    # print(f'Decision: {decision_list}')
                    Indi_Demand = np.dot(usedDemand, range(self.I))

                    if decision_list:
                        mylist.append(1)
                        current_loss += 2
                        total_loss += 1
                        # find the row can assign usedDemand（j)
                        for k, pattern in enumerate(newx):
                            if pattern[decision_list] > 0 and change_roll[k] > (self.I + 1):
                                newx[k][decision_list] -= 1
                                if decision_list - Indi_Demand - 2 >= 0:
                                    newx[k][int(decision_list -
                                                Indi_Demand - 2)] += 1
                                change_roll[k] -= (Indi_Demand+2)
                                break
                            if k == len(newx)-1:
                                for j, pat in enumerate(newx):
                                    if pat[decision_list] > 0:
                                        newx[j][decision_list] -= 1
                                        if decision_list - Indi_Demand - 2 >= 0:
                                            newx[j][int(
                                                decision_list - Indi_Demand - 2)] += 1
                                        change_roll[j] -= (Indi_Demand+2)
                                        break

                    else:
                        mylist.append(0)
                    remaining_period -= 1
                elif any(usedDemand) == 0:
                    mylist.append(0)
                    remaining_period -= 1
            else:
                if any(usedDemand) == 0:
                    usedDemand, decision_list = decisionOnce(
                        sequence, demand, self.probab)
                    # print(f'Decision: {decision_list}')
                    Indi_Demand = np.dot(usedDemand, range(self.I))

                    if decision_list:
                        mylist.append(1)
                        current_loss += 2
                        total_loss += 1
                        # find the row can assign usedDemand（j)
                        for k, pattern in enumerate(newx):
                            if pattern[decision_list] > 0 and change_roll[k] > (self.I + 1):
                                newx[k][decision_list] -= 1
                                if decision_list - Indi_Demand - 2 >= 0:
                                    newx[k][int(decision_list -
                                                Indi_Demand - 2)] += 1
                                change_roll[k] -= (Indi_Demand+2)
                                break
                            if k == len(newx)-1:
                                for j, pat in enumerate(newx):
                                    if pat[decision_list] > 0:
                                        newx[j][decision_list] -= 1
                                        if decision_list - Indi_Demand - 2 >= 0:
                                            newx[j][int(
                                                decision_list - Indi_Demand - 2)] += 1
                                        change_roll[j] -= (Indi_Demand+2)
                                        break

                    else:
                        mylist.append(0)
                    remaining_period -= 1

            remaining_period0 = remaining_period
            sequence = sequence[-remaining_period:]

            total_usedDemand += usedDemand

            # use deterministic calculate
            ini_demand, newx = deterModel.IP_formulation2(
                change_roll, remaining_period0, self.probab)
            newx = newx.T.tolist()

        sequence1 = [i-1 for i in sequence1 if i > 0]

        final_demand1 = np.array(sequence1) * np.array(mylist)
        final_demand1 = final_demand1[final_demand1 != 0]

        demand = np.zeros(self.I)
        for i in final_demand1:
            demand[i-1] += 1
        return demand

    def method5(self, sequence, ini_demand, newx, change_roll0):
        # newx patterns for N rows.
        change_roll = copy.deepcopy(change_roll0)
        newx = newx.T.tolist()
        mylist = []
        remaining_period0 = self.num_period
        sequence1 = copy.copy(sequence)
        total_usedDemand = np.zeros(self.I)
        # ini_demand1 = np.array(self.probab) * self.num_period
        # deterModel = deterministicModel(
            # self.roll_width, self.given_lines, self.demand_width_array, self.I)

        while remaining_period0:
            demand = ini_demand

            usedDemand, remaining_period = decisionSeveral(sequence, demand)

            diff_period = remaining_period0 - remaining_period

            demand_list = sequence[0:diff_period]

            for j in demand_list:
                for k, pattern in enumerate(newx):
                    if pattern[j-2] > 0 and (change_roll[k] > (self.I + 1) or change_roll[k] == j):
                        newx[k][j-2] -= 1
                        change_roll[k] -= j
                        break

                    if k == len(newx)-1:
                        # for t, i in enumerate(change_roll):
                        #     newx[t][i-2] = 1
                        for kk, pat in enumerate(newx):
                            if pat[j-2] > 0:
                                newx[kk][j-2] -= 1
                                change_roll[kk] -= j
                                break

            mylist += [1] * diff_period

            if any(usedDemand) == 0:  # all are 0
                usedDemand, decision_list = decisionOnce(
                    sequence, demand, self.probab)
                # print(f'Decision: {decision_list}')
                Indi_Demand = np.dot(usedDemand, range(self.I))

                if decision_list:
                    mylist.append(1)

                    # find the row can assign usedDemand（j)
                    for k, pattern in enumerate(newx):
                        if pattern[decision_list] > 0 and change_roll[k] > (self.I + 1):
                            newx[k][decision_list] -= 1
                            if decision_list - Indi_Demand - 2 >= 0:
                                newx[k][int(decision_list -
                                            Indi_Demand - 2)] += 1
                            change_roll[k] -= (Indi_Demand+2)
                            break
                        if k == len(newx)-1:
                            for j, pat in enumerate(newx):
                                if pat[decision_list] > 0:
                                    newx[j][decision_list] -= 1
                                    if decision_list - Indi_Demand - 2 >= 0:
                                        newx[j][int(
                                            decision_list - Indi_Demand - 2)] += 1
                                    change_roll[j] -= (Indi_Demand+2)
                                    break

                else:
                    mylist.append(0)
                remaining_period -= 1


            remaining_period0 = remaining_period
            sequence = sequence[-remaining_period:]

            total_usedDemand += usedDemand

            sam = samplingmethod(I, num_sample, remaining_period0, probab)
            dw, prop = sam.get_prob()
            W = len(dw)

            
            m1 = stochasticModel(change_roll, self.given_lines,
                                 self.demand_width_array, W, self.I, prop, dw)

            ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)
            
            # #  use stochastic calculate
            deterModel = deterministicModel(
                change_roll, self.given_lines, self.demand_width_array, self.I)
            ini_demand, _ = deterModel.IP_formulation(
                np.zeros(self.I), ini_demand)
            ini_demand, newx = deterModel.IP_formulation(ini_demand, np.zeros(self.I))

            newx = newx.T.tolist()

        sequence1 = [i-1 for i in sequence1 if i > 0]

        final_demand1 = np.array(sequence1) * np.array(mylist)
        final_demand1 = final_demand1[final_demand1 != 0]

        demand = np.zeros(self.I)
        for i in final_demand1:
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

    def result(self, sequence, ini_demand, ini_demand3, newx3, newx4):
        ini_demand4 = copy.deepcopy(ini_demand)
        roll_width = copy.deepcopy(self.roll_width)

        final_demand1 = self.method1(sequence, ini_demand)
        # final_demand1 = 0

        final_demand3 = self.method6(sequence, ini_demand3, newx3, roll_width)
        # final_demand3 = 0
        final_demand4 = self.method6(sequence, ini_demand4, newx4, roll_width)

        return final_demand1, final_demand3, final_demand4


def prop_list():
    x = np.arange(0.05, 1, 0.1)
    y = np.arange(0.05, 0.8, 0.1)
    p = np.zeros((len(x)*len(y), 4))

    t = 0
    for i in x:
        for j in y:
            if 3-2*i-4*j > 0 and 3-4*i-2*j > 0:
                p[t] = [(3 - 4*i - 2*j)/6, i, j, (3 - 2*i - 4*j)/6]
                t += 1
    p = p[0:t]

    return p

def prop_list1():
    x = np.arange(0.05, 0.5, 0.1)  # p3
    y = np.arange(0.05, 0.35, 0.05)  # p4
    p = np.zeros((len(x)*len(y), 4))

    t = 0
    for i in x:
        for j in y:
            if 1-2*i-3*j > 0:
                p[t] = [(i + 2*j), (1 - 2*i - 3*j), i, j]
                t += 1
    p = p[0:t]

    return p


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 80
    given_lines = 10
    # np.random.seed(i)
    p = prop_list()

    begin_time = time.time()
    filename = 'Results_' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    for probab in p:
        my_file.write('probabilities: \t' + str(probab) + '\n')
        # probab = [0.3, 0.5, 0.1, 0.1]
        roll_width = np.ones(given_lines) * 21
        # total_seat = np.sum(roll_width)

        a_instance = CompareMethods(
            roll_width, given_lines, I, probab, num_period, num_sample)

        ratio1 = 0
        ratio2 = 0
        ratio3 = 0
        ratio4 = 0
        ratio5 = 0
        ratio6 = 0
        ratio7 = 0
        accept_people = 0
        num_people = 0

        multi = np.arange(1, I+1)

        count = 50
        for j in range(count):
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()

            total_people = sum(sequence) - num_period

            a, c, d = a_instance.result(sequence, ini_demand, ini_demand3, newx3, newx4)

            b = a_instance.dynamic_program(sequence)

            e = a_instance.row_by_row(sequence)
            baseline = np.dot(multi, e)

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            h = a_instance.bid_price(sequence)

            seq = a_instance.binary_search_first(sequence)

            g = a_instance.offline(seq)
            # ratio1 += (np.dot(multi, a)-baseline)/ baseline

            ratio1 += np.dot(multi, a) / optimal
            ratio2 += np.dot(multi, b) / optimal
            ratio3 += np.dot(multi, c) / optimal
            ratio4 += np.dot(multi, d) / optimal
            ratio5 += np.dot(multi, e) / optimal
            ratio6 += np.dot(multi, g) / optimal
            ratio7 += np.dot(multi, h) / optimal
            accept_people += optimal
            num_people += total_people

        my_file.write('M1: %.2f ;' % (ratio1/count*100))
        my_file.write('Dy: %.2f ;' % (ratio2/count*100))
        my_file.write('Mean: %.2f ;' % (ratio3/count*100))
        my_file.write('Sto: %.2f ;' % (ratio4/count*100))
        my_file.write('FCFS: %.2f ;' % (ratio5/count*100))
        my_file.write('FCFS1: %.2f ;' % (ratio6/count*100))
        my_file.write('bid-price: %.2f \n' % (ratio7/count*100))
        my_file.write('Number of accepted people: %.2f \t' %
                      (accept_people/count))
        my_file.write('Number of people: %.2f \n' % (num_people/count))
        # f.write(str(ratio6/count*100) + '\n')

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t%f\n' % run_time)
    # print('%.2f' % (ratio1/count*100))
    # print('%.2f' % (ratio2/count*100))
    # print('%.2f' % (ratio3/count*100))
    # print('%.2f' % (ratio4/count*100))
    # print('%.2f' % (ratio5/count*100))
    # print('%.2f' % (ratio6/count*100))
    my_file.close()
