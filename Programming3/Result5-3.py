# Impact of social distance under different demands
import numpy as np
from Comparison import CompareMethods
import matplotlib.pyplot as plt
import copy
import time
# Without social distance: 累加 直到capacity
# x-axis: number of people
# gap point
# Group graphs

def withoutSD(sequence, total_seat, sd):
    accept_people = 0
    sequence = [i-sd for i in sequence]

    for i in sequence:
        if accept_people + i <= total_seat:
            accept_people += i
    return accept_people

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(30,100,1)
    given_lines = 10
    # np.random.seed(i)
    sd = 1
    p = [[0.25, 0.25, 0.25, 0.25], [0.4, 0.4, 0.1, 0.1]]
    # probab = [0.4, 0.2, 0.3, 0.1]
    #  [0.4, 0.4, 0.1, 0.1]

    begin_time = time.time()
    filename = 'different_c' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    t_value = np.arange(30, 100, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))
    occup_without = np.zeros(len(period_range))

    for probab in p:
        gamma = np.dot(probab, np.arange(1,5))
        cnt = 0
        gap_if = True
        for num_period in period_range:
            roll_width = np.ones(given_lines) * 21
            # total_seat = np.sum(roll_width)
            total_seat = np.sum(roll_width) - given_lines * sd

            a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)

            sto = 0
            accept_people = 0

            multi = np.arange(1, I+1)
            count = 50
            for j in range(count):
                sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
                sequence1 = copy.deepcopy(sequence)

                g = a_instance.method_new(sequence1, newx4, roll_width)
                sto += np.dot(multi, g)

                accept_people += withoutSD(sequence, total_seat, sd)

            occup_value[cnt] = sto/count/total_seat * 100
            people_value[cnt] = accept_people/count/total_seat * 100
            if gap_if:
                if accept_people/count - sto/count > 1:
                    point = [num_period-1, occup_value[cnt-1]]
                    gap_if = False
            cnt += 1

        plt.plot(t_value* gamma, people_value, 'b-', label = 'Without social distancing')
        plt.plot(t_value* gamma, occup_value, 'r--', label = 'With social distancing')
        plt.xlim((50, 250))
        plt.ylim((0, 100))
        plt.xlabel('Expected number of people')
        plt.ylabel('Percentage of total seats')
        point[1] = round(point[1], 2)
        # plt.annotate(r'Gap $%s$' % str(point), xy=point, xytext=(
            # point[0]*gamma + 10, point[1]-20), arrowprops=dict(facecolor='black', shrink=0.1),)
        
        plt.annotate(r'Gap $%s$' % str(point), xy= (point[0]*gamma, point[1]), xytext=(point[0]*gamma + 10, point[1]-20), arrowprops=dict(facecolor='black', shrink=0.1),)

        my_x_ticks = np.arange(50, 250, 50)
        plt.xticks(my_x_ticks)
        plt.legend()
        graphname = './test' + str(probab[0]) + '.pdf'
        plt.savefig(graphname)
        plt.cla()

        for i in range(60, 100, 10):
            a = people_value[i- period_range[0]] - occup_value[i - period_range[0]]
            my_file.write(str(i) + '\t' + str(a) + '\n')

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t %f \n' % run_time)
    my_file.close()