import numpy as np
from Comparison import CompareMethods
import matplotlib.pyplot as plt

# This function cals Impact of Social Distancing as Demand Increase


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4   # the number of group types
    period_range = range(10,100,1)
    given_lines = 10
    sd = 1
    # np.random.seed(i)
    probab1 = [[0.3, 0.2, 0.2, 0.3]]
    # probab1 = [[0.2, 0.3, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25], [0.3, 0.2, 0.2, 0.3], [0.1, 0.4, 0.4, 0.1], [0.1, 0.5, 0.2, 0.2]]

    test = 0
    for probab in probab1:
        t_value = np.arange(10, 100, 1)
        people_value = np.zeros(len(period_range))
        occup_value = np.zeros(len(period_range))

        cnt = 0
        gap_if = True
        for num_period in period_range:
            roll_width = np.ones(given_lines) * 21
            total_seat = np.sum(roll_width)

            a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)

            M4 = 0
            accept_people = 0

            multi = np.arange(1, I+1)
            print(num_period)
            count = 50
            for j in range(count):
                sequence, ini_demand, ini_demand3, _, _ = a_instance.random_generate()

                # a, c, d = a_instance.result(sequence, ini_demand, ini_demand3)

                f = a_instance.offline(sequence)  # optimal result
                optimal = np.dot(multi, f)

                d = a_instance.offline(sequence)

                # sequence, ini_demand = a_instance.random_generate()
                # a = a_instance.result(sequence, ini_demand)

                M4 += np.dot(multi, d)
                accept_people += optimal

            occup_value[cnt] = M4/count/total_seat * 100
            people_value[cnt] = accept_people/count/total_seat * 100
            if gap_if:
                if accept_people/count - M4/count > 1:
                    point = [num_period-1, occup_value[cnt-1]]
                    gap_if = False
            cnt += 1
        if test <= 0:
            plt.plot(t_value, people_value, 'b-', label='Without social distancing')
            test += 1
        plt.plot(t_value, occup_value, label = str(probab))
        point[1] = round(point[1], 2)
        plt.annotate(r'Gap $%s$' % str(point), xy=point, xytext=(point[0]+10, point[1]-20), arrowprops=dict(facecolor='black', shrink=0.1),)
        print(point)
    plt.xlabel('Periods')
    plt.ylabel('Percentage of total seats')
    plt.legend()
    plt.show()
        

