import numpy as np
from Comparison import CompareMethods
import matplotlib.pyplot as plt
import copy
import time
import random

# Obtain the gap point of 200 probabilities.

# Find all the probability combinations

def prop_all():
    x2 = np.arange(0.05, 1, 0.05)
    x3 = np.arange(0.05, 1, 0.05)
    x4 = np.arange(0.05, 1, 0.05)

    p = np.zeros((len(x2)*len(x3)*len(x4), 4))
    t = 0

    for i in x2:
        for j in x3:
            for k in x4:
                if 1 - i - j - k > 0:
                    p[t] = [1 - i - j - k, i, j, k]
                    t += 1

    p = p[0:t]

    return p



if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(45,80,1)
    given_lines = 10
    sd = 1
    probab = prop_all()
    all_number = len(probab)

    begin_time = time.time()
    filename = 'Periods_' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    t_value = np.arange(45, 80, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))

    for i in range(200):
        ran_prop = random.randint(0, all_number)
        p = probab[ran_prop]
        my_file.write(str(p) + '\t')
        gap_if = True
        cnt = 0
        for num_period in period_range:
            roll_width = np.ones(given_lines) * 21
            total_seat = np.sum(roll_width) - given_lines * sd
            a_instance = CompareMethods(roll_width, given_lines, I, p, num_period, num_sample, sd)
            a_without = CompareMethods(roll_width-1, given_lines, I, p, num_period, num_sample, 0)
            sto = 0
            accept_people = 0

            multi = np.arange(1, I+1)
            count = 50
            for j in range(count):
                sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
                sequence1 = [i-sd for i in sequence]
                f = a_without.offline(sequence1)  # optimal result
                optimal = np.dot(multi, f)

                g = a_instance.method_new(sequence, newx4, roll_width)
                sto += np.dot(multi, g)
                accept_people += optimal

            occup_value[cnt] = sto/count/total_seat * 100
            people_value[cnt] = accept_people/count/total_seat * 100
            if gap_if:
                if accept_people/count - sto/count > 1:
                    point = [num_period-1, occup_value[cnt-1]]
                    my_file.write(str(point) + '\n')
                    gap_if = False
                    break
            cnt += 1
        print(cnt)
    my_file.close()
