# Impact of social distance under different demands
import numpy as np
from Comparison import CompareMethods
import matplotlib.pyplot as plt

# The occupancy rate with different seat layouts
# 11 * 15-25  the fan-shaped layout
# 15 * 8 small room
# 10 * 21 big room
# 20 * 11

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(40, 100, 1)
    given_lines = 11
    # np.random.seed(i)
    probab = [0.3, 0.2, 0.2, 0.3]
    sd = 1
    t_value = np.arange(40, 100, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))
    occup_without = np.zeros(len(period_range))
    cnt = 0
    gap_if = True
    for num_period in period_range:
        print(num_period)
        # roll_width = np.ones(given_lines) * 11
        roll_width = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        total_seat = np.sum(roll_width) - given_lines * sd

        a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)
        a_without = CompareMethods(roll_width-sd, given_lines, I, probab, num_period, num_sample, 0)
        sto = 0
        # sto_without = 0
        accept_people = 0

        multi = np.arange(1, I+1)

        count = 1
        for j in range(count):
            sequence, newx4 = a_instance.random_generate()
            sequence1 = [i-sd for i in sequence]
            _, newx40 = a_without.random_generate()
            f = a_without.offline(sequence1)  # optimal result
            optimal = np.dot(multi, f)

            # d = a_instance.offline(sequence)
            # g0 = a_without.method_new(sequence1, newx40, roll_width-sd)
            # sto_without += np.dot(multi, g0)

            g = a_instance.method_new(sequence, newx4, roll_width)
            sto += np.dot(multi, g)
            accept_people += optimal
            
        occup_value[cnt] = sto/count/total_seat * 100
        # occup_without[cnt] = sto_without/count/total_seat * 100
        people_value[cnt] = accept_people/count/total_seat * 100
        if gap_if:
            if accept_people/count - sto/count > 1:
                point = [num_period-1, occup_value[cnt-1]]
                gap_if = False
        cnt += 1
    plt.plot(t_value, people_value, 'b-', label='With 0 social distancing')
    plt.plot(t_value, occup_value, 'r--', label='With 1 social distancing')
    # plt.plot(t_value, occup_without, 'y*', label='Dynamic Without SD')
    plt.xlabel('Periods')
    plt.ylabel('Percentage of total seats')
    point[1] = round(point[1], 2)
    plt.annotate(r'Gap $%s$' % str(point), xy = point, xytext = (point[0]+10, point[1]-20), arrowprops = dict(facecolor='black', shrink=0.1),)
    plt.legend()
    plt.savefig('test.pdf')
    print(point)
