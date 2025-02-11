from collections import Counter
from scipy.stats import binom
import numpy as np
import copy

def several_class(size_group, demand, remaining_period, probab, sd):
    # The function is used to give the maximum difference and give the decision
    # j F(x_j -1, T, p_j) - (j-i-\delta) F(x_{j-i-\delta}, T, p_{j-i-\delta}) -\delta
    # here probab should be a vector of arrival rate
    # demand is the current left demand
    # size_group is the actual size of group i
    #  return 0-index
    delta = sd
    max_size = len(demand)
    if size_group == max_size:
        return False
    diff_set = np.zeros(max_size - size_group)
    count = 0
    for j in range(size_group+1, max_size+1, 1):
        term1 = j * binom.cdf(demand[j-1]-1, remaining_period, probab[j-1])
        term2 = (j- size_group - delta) * \
            binom.cdf(demand[j-size_group-delta-1], remaining_period, probab[j-size_group-delta-1])
        if j <= size_group + delta:
            diff_set[count] = size_group - j + term1
        else:
            diff_set[count] = term1 - term2 - delta
            # i -j + term1 if j < i + delta
        count += 1
    max_diff = max(diff_set)
    index_diff = np.argmax(diff_set) + size_group

    if max_diff > 0:
        return index_diff
    # if diff_set[0] > 0:
    #     return size_group
    else:
        return False

def decision1(sequence, demand, probab, sd):
    # the function is used to make a decision on several classes
    # sequence is one possible sequence of the group arrival.
    period = len(sequence)
    I = len(demand)
    group_type = [i+1+sd for i in range(I)]
    decision_list = [0] * period
    t = 0
    for i in sequence:
        remaining_period = period - t
        position = i -1 - sd
        demand_posi = demand[position]

        if demand_posi > 0:
            decision_list[t] = 1
            demand[position] = demand_posi - 1
        elif sum(demand) < 1e-4:
            break
        elif i == group_type[-1] and demand[-1] == 0:
            decision_list[t] = 0
        else:
            accept_reject = several_class(i-sd, demand, remaining_period-1, probab, sd)
            if accept_reject:
                decision_list[t] = 1
                demand[accept_reject] -= 1
                if accept_reject-position-1-sd >= 0:
                    demand[accept_reject-position-1-sd] += 1
        t += 1
    return decision_list

def generate_sequence(period, prob, sd):
    I = len(prob)
    group_type = np.arange(1+ sd, 1+ sd+ I)
    trials = [np.random.choice(group_type, p = prob) for _ in range(period)]
    return trials

def sequence_pool(count, num_period, probab, s):
    pools = np.zeros((count, num_period), dtype = int)
    for i in range(count):
        pools[i] = generate_sequence(num_period, probab, s)
    return pools

def decisionOnce(sequence, demand0, probab, sd):
    # the function is used to make a decision once on several classes
    # sequence is one possible sequence of the group arrival.
    # decision_list is the index of the larger group
    # record_demand is the index of request
    demand = copy.deepcopy(demand0)
    I = len(demand)
    record_demand = np.zeros(I)
    period = len(sequence)
    group_type = [i+1+ sd for i in range(I)]
    decision_list = 0
    i = sequence[0]
    remaining_period = period
    position = i -1 -sd

    if i == group_type[-1] and demand[-1] == 0:
        decision_list = 0
    else:
        accept_reject = several_class(i-sd, demand, remaining_period-1, probab, sd)
        if accept_reject:
            decision_list = accept_reject
            demand[accept_reject] -= 1
            if accept_reject-position-1-sd >= 0:
                demand[accept_reject-position-1-sd] += 1
            record_demand[position] = 1
    return record_demand, decision_list

def decisionSeveral(sequence, demand):
    # The function is used to make several decisions
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

def decision2(sequence, demand):
    # The function is used to make a decision based on seat planning
    # sequence is one possible sequence of the group arrival.
    period = len(sequence)
    I = len(demand)
    group_type = [i+2 for i in range(I)]
    decision_list = [0] * period
    t = 0
    for i in sequence:
        position = group_type.index(i)
        demand_posi = demand[position]
        if demand_posi > 0:
            decision_list[t] = 1
            demand[position] = demand_posi - 1
        elif i == group_type[-1] and demand[-1] == 0:
            print('It is full.')
            break
        else:
            decision_list[t] = 0
        t += 1
        # print('the period:', t)
        # print('demand is:', demand)
    return t, decision_list[0:t]

# def decisionSeveral(sequence, demand):
#     # the function is used to make several decisions
#     # Sequence is one possible sequence of the group arrival.
#     period = len(sequence)
#     group_type = sorted(list(set(sequence)))
#     originalDemand = copy.deepcopy(demand)
#     t = 0
#     for i in sequence:
#         remaining_period = period - t
#         position = group_type.index(i)
#         demand_posi = demand[position]
#         if demand_posi > 0:
#             # decision_list[t] = 1
#             demand[position] = demand_posi - 1
#         else:
#             usedDemand = originalDemand - demand
#             break
#         t += 1
#     # decision_list = decision_list[0:t]
#     return usedDemand, remaining_period
