# Impact of social distance under different demands
import numpy as np
from Comparison0 import CompareMethods0
from Comparison import CompareMethods
import matplotlib.pyplot as plt
import copy
import time
# Without social distance: 累加 直到capacity
# x-axis: number of people
# Difference at specific periods

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
    # period_range = range(30,100,1)
    given_lines = 10
    # np.random.seed(i)
    # p = [[0.2, 0.2, 0.3, 0.3], [0.25, 0.25, 0.25, 0.25], [0.3, 0.3, 0.2, 0.2], [0.4, 0.2, 0.3, 0.1], [0.4, 0.4, 0.1, 0.1]]
    p = [[0.45, 0.35, 0.05, 0.15], [0.35, 0.35, 0.15, 0.15], [0.35, 0.25, 0.15, 0.25], [0.3, 0.2, 0.2, 0.3], [0.25, 0.15, 0.25, 0.35]]
    # probab = [0.4, 0.2, 0.3, 0.1]
    #  [0.4, 0.4, 0.1, 0.1]

    begin_time = time.time()
    filename = 'different_c' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    t_value = np.arange(30, 100, 1)
    people_value = np.zeros(5)
    occup_value = np.zeros(5)
    occup_without = np.zeros(5)

    people_range = np.array([130, 150, 170, 190, 210])
    for probab in p:
        gamma = np.dot(probab, np.arange(1,5))
        cnt = 0
        gap_if = True
        my_file.write(str(probab) + '\n')
        period_range = np.around(people_range/gamma)
        period_range = [int(i) for i in period_range]

        for num_period in period_range:
            roll_width = np.ones(given_lines) * 21
            # total_seat = np.sum(roll_width)
            total_seat = np.sum(roll_width) - given_lines

            a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)

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

            occup_value[cnt] = sto/count/total_seat * 100
            people_value[cnt] = accept_people/count/total_seat * 100

            cnt += 1
        diff = people_value - occup_value
        my_file.write(str(people_value) + '\t' + str(occup_value) + '\n')
        my_file.write(str(diff) + '\n')

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t %f \n' % run_time)
    my_file.close()