import numpy as np
from Comparison0 import CompareMethods0
from Comparison import CompareMethods
import matplotlib.pyplot as plt
import copy
import time
import random
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


num_sample = 1000  # the number of scenarios
I = 4  # the number of group types
given_lines = 10
probab = prop_all()
all_number = len(probab)

begin_time = time.time()

filename = 'Periods_' + str(time.time()) + '.txt'
my_file = open(filename, 'w')
my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

for i in range(100):
    print(i)
    ran_prop = random.randint(0, all_number-1)
    p = probab[ran_prop]
    my_file.write(str(p) + '\t')
    gamma = np.dot(p, np.arange(1, 5))
    a = 25
    b = 50
    num_period = int((a+b)/2)
    while num_period > a:
        roll_width = np.ones(given_lines) * 11
        # total_seat = np.sum(roll_width)
        total_seat = np.sum(roll_width) - given_lines
        a_instance = CompareMethods(roll_width, given_lines, I, p, num_period, num_sample)
        a_without = CompareMethods0(roll_width-1, given_lines, I, p, num_period, num_sample, 0)
        sto = 0
        accept_people = 0

        multi = np.arange(1, I+1)
        count = 100
        for j in range(count):
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
# [5, 5, 3, 5, 2, 5, 5, 5, 2, 5, 2, 3, 5, 5, 5, 5, 5, 5, 5, 2, 5, 5, 5, 2, 5,
    # 5, 5, 5, 5, 5, 5, 2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 5, 5, 5, 5, 5, 2, 5, 2]
