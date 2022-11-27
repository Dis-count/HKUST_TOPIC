from collections import Counter
from scipy.stats import binom
import numpy as np
import copy


def several_class(size_group, demand, remaining_period, probab):
    # The function is used to give the maximum difference and give the decision
    # j F(x_j -1, T, p_j) - (j-i-1) F(x_{j-i-1}, T, p_{j-i-1}) -1
    # here probab should be a vector of arrival rate
    # demand is the current left demand
    # size_group is the actual size of group i
    max_size = len(demand)
    if size_group == max_size:
        return False
    diff_set = np.zeros(max_size - size_group)
    count = 0
    for j in range(size_group+1, max_size+1, 1):
        term1 = j * binom.cdf(demand[j-1]-1, remaining_period, probab[j-1])
        term2 = (j-size_group-1) * \
            binom.cdf(demand[j-size_group-2], remaining_period, probab[j-size_group-2])
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
    I = len(demand)
    group_type = [i+2 for i in range(I)]
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
        elif i == group_type[-1] and demand[-1] == 0:
            decision_list[t] = 0
        else:
            accept_reject = several_class(
                i-1, demand, remaining_period-1, probab)
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
    I = len(prob)
    group_type = np.arange(2, 2 + I)
    trials = [np.random.choice(group_type, p=prob) for _ in range(period)]
    return trials


def decision_demand(sequence, decision_list):
    accept_list = np.multiply(sequence, decision_list)
    dic = Counter(accept_list)
    # Sort the list according to the value of dictionary.
    res_demand = [dic[key] for key in sorted(dic)]
    return res_demand


def decisionOnce(sequence, demand, probab):
    # the function is used to make a decision once on several classes
    # sequence is one possible sequence of the group arrival.
    I = len(demand)
    record_demand = np.zeros(I)
    period = len(sequence)
    group_type = [i+2 for i in range(I)]
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
    I = len(demand)
    group_type = [i+2 for i in range(I)]

    # decision_list = [0] * period
    originalDemand = copy.deepcopy(demand)
    t = 0
    for i in sequence:
        position = group_type.index(i)
        demand_posi = demand[position]
        if demand_posi > 0:
            # decision_list[t] = 1
            demand[position] = demand_posi - 1
        else:
            remaining_period = period - t
            break
        t += 1
        remaining_period = period - t
    usedDemand = originalDemand - demand
        # print('the period:', t)
        # print('demand is:', demand)
    # decision_list = decision_list[0:t]
    return usedDemand, remaining_period

