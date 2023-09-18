# Impact of social distance under different demands
import numpy as np
from Comparison0 import CompareMethods0
from Comparison import CompareMethods
import matplotlib.pyplot as plt
import copy

# Without social distance: 累加 直到capacity
# x-axis: number of people
#  individual graph

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
    period_range = range(30,100,1)
    given_lines = 10
    # np.random.seed(i)
    probab = [0.3, 0.3, 0.2, 0.2]
    # probab = [0.3, 0.2, 0.2, 0.3]
    gamma = np.dot(probab, np.arange(1,5))

    t_value = np.arange(30, 100, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))
    occup_without = np.zeros(len(period_range))
    cnt = 0
    gap_if = True
    for num_period in period_range:
        roll_width = np.ones(given_lines) * 21
        # total_seat = np.sum(roll_width)
        total_seat = np.sum(roll_width) - given_lines

        a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)

        sto = 0
        accept_people = 0

        multi = np.arange(1, I+1)
        print(num_period)
        count = 50
        for j in range(count):
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
            sequence1 = copy.deepcopy(sequence)
            sequence0 = [i-1 for i in sequence]

            g = a_instance.method_new(sequence1, newx4, roll_width)
            sto += np.dot(multi, g)

            accept_people += withoutSD(sequence, total_seat)

        occup_value[cnt] = sto/count/total_seat * 100
        people_value[cnt] = accept_people/count/total_seat * 100
        if gap_if:
            if accept_people/count - sto/count >= 1:
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
    plt.savefig('./test2.pdf')
    plt.show()