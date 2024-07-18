import numpy as np
from Comparison0 import CompareMethods0
from Comparison import CompareMethods
import matplotlib.pyplot as plt
import copy
import time
import random

# Find all the probability combinations
#  the average of random probability with the same gamma

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


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    given_lines = 10

    begin_time = time.time()
    filename = 'Periods_' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')
    lengths = [36]
    for length in lengths:
        my_file.write(str(length) + '\n')
        for gamma in np.arange(1.5, 3.5, 0.1):
            print(gamma)
            probab = prop_list(gamma)
            all_number = len(probab)

            my_file.write(str(gamma) + '\n')
            a = 70
            b = 160
            num_period = int((a+b)/2)
            while num_period > a:
                roll_width = np.ones(given_lines) * length
                total_seat = np.sum(roll_width) - given_lines
                sto = 0
                accept_people = 0
                multi = np.arange(1, I+1)
                count = 100

                for j in range(count):
                    ran_prop = random.randint(0, all_number-1)
                    p = probab[ran_prop]
                    a_instance = CompareMethods(roll_width, given_lines, I, p, num_period, num_sample)
                    a_without = CompareMethods0(roll_width-1, given_lines, I, p, num_period, num_sample, 0)
                    sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
                    sequence1 = copy.deepcopy(sequence)
                    f = a_without.offline(sequence)  # optimal result
                    optimal = np.dot(multi, f)
                    g = a_instance.method_new(sequence1, newx4, roll_width)
                    sto += np.dot(multi, g)
                    accept_people += optimal

                occup_value = sto/count/total_seat * 100

                if accept_people/count - sto/count > 1:
                    b = num_period
                    num_period = int((a + b)/2)
                else:
                    a = num_period
                    num_period = int((a + b)/2)

            point = [num_period, occup_value]
            my_file.write(str(point) + '\n')

    my_file.close()
