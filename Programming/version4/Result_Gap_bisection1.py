import numpy as np
from Comparison0 import CompareMethods0
from Comparison import CompareMethods
import matplotlib.pyplot as plt
import copy
import time
import random
# Find the gap point by bisection

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
    probab = [[0.45, 0.35, 0.05, 0.15], [0.35, 0.35, 0.15, 0.15], [0.35, 0.25, 0.15, 0.25], [0.3, 0.2, 0.2, 0.3], [0.25, 0.15, 0.25, 0.35]]

    begin_time = time.time()
    filename = 'Periods_prop25' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    for p in probab:
        gamma = np.dot(p, np.arange(1,5))
        my_file.write(str(p) + '\t')
        my_file.write(str(gamma) + '\t')
        a = 40
        b = 90
        num_period = int((a+b)/2)
        while num_period > a:
            # roll_width = np.ones(given_lines) * 21
            roll_width = np.array([21, 21, 21, 21, 21, 21, 21, 21, 21, 21])
            total_seat = np.sum(roll_width) - given_lines
            a_instance = CompareMethods(roll_width, given_lines, I, p, num_period, num_sample)
            sto = 0
            accept_people = 0

            multi = np.arange(1, I+1)
            count = 100
            for j in range(count):
                sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
                sequence1 = copy.deepcopy(sequence)

                g = a_instance.method_new(sequence1, newx4, roll_width)
                sto += np.dot(multi, g)

                accept_people += withoutSD(sequence, total_seat)
            occup_value = sto/count/total_seat * 100
            people_value = accept_people/count/total_seat * 100

            if people_value/count - sto/count > 1:
                b = num_period
                num_period = int((a + b)/2)
            else:
                a = num_period
                num_period = int((a + b)/2)

        point = [num_period, occup_value]
        my_file.write(str(point) + '\n')

    my_file.close()
