# Impact of social distance under different demands
import numpy as np
from Comparison0 import CompareMethods0
from Comparison import CompareMethods
import matplotlib.pyplot as plt
import copy
import time
import random
# Without social distance: 累加 直到capacity
# x-axis: number of people

# Difference at specific periods under different gamma

# gamma = 2
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


def prop_list(gamma):
    x = np.arange(0.05, 1, 0.05)
    y = np.arange(0.05, 1, 0.05)
    p = np.zeros((len(x)*len(y), 4))

    t = 0
    for i in x:
        for j in y:
            if 4 - gamma - 3*i - 2*j > 0 and 2*i + j - 3 + gamma > 0:
                p[t] = [i, j, (4 - gamma - 3*i - 2*j), (2*i + j - 3 + gamma)]
                t += 1
    p = p[0:t]

    return p

def withoutSD(sequence, total_seat):
    accept_people = 0
    sequence = [i-1 for i in sequence]

    for i in sequence:
        if accept_people + i <= total_seat:
            accept_people += i
    return accept_people


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    given_lines = 10
    p = prop_list(2)
    all_number = len(p)

    begin_time = time.time()
    filename = 'different_c' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    people_value = np.zeros(5)
    occup_value = np.zeros(5)
    occup_without = np.zeros(5)

    people_range = np.array([130, 150, 170, 190, 210])
    
    cnt = 0
    gamma = 2
    period_range = np.around(people_range/gamma)
    period_range = [int(i) for i in period_range]

    for num_period in period_range:
        roll_width = np.ones(given_lines) * 21

        total_seat = np.sum(roll_width) - given_lines
        count = 100
        sto = 0
        accept_people = 0
        for j in range(count):
            ran_prop = random.randint(0, all_number-1)
            probab = p[ran_prop]
            a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)

            multi = np.arange(1, I+1)
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
            sequence1 = copy.deepcopy(sequence)

            g = a_instance.method_new(sequence1, newx4, roll_width)
            sto += np.dot(multi, g)

            accept_people += withoutSD(sequence, total_seat)

        occup_value[cnt] = sto/count/total_seat * 100
        people_value[cnt] = accept_people/count/total_seat * 100

        cnt += 1
    diff = people_value - occup_value
    my_file.write(str(people_value) + '\t' + str(occup_value) + '\n')
    my_file.write(str(diff) + '\n')

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t %f \n' % run_time)
    my_file.close()
