# Impact of social distance under different demands
import numpy as np
from Comparison import CompareMethods
import matplotlib.pyplot as plt

# The occupancy rate with different social distancings over periods

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(30,100,1)
    given_lines = 10
    # np.random.seed(i)
    sd = 1
    probab = [0.3, 0.2, 0.2, 0.3]

    t_value = np.arange(30, 100, 1)
    people_value = np.zeros(len(period_range))
    occup_1 = np.zeros(len(period_range))
    occup_2 = np.zeros(len(period_range))

    cnt = 0

    for num_period in period_range:
        roll_width = np.ones(given_lines) * 21
        # total_seat = np.sum(roll_width)
        total_seat = np.sum(roll_width) - given_lines * sd

        a_0 = CompareMethods(roll_width - 1, given_lines, I, probab, num_period, num_sample, 0)
        a_1 = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)
        a_2= CompareMethods(roll_width + 1, given_lines, I, probab, num_period, num_sample, sd+1)

        sto_1 = 0
        sto_2 = 0
        accept_people = 0

        multi = np.arange(1, I+1)

        count = 100
        for j in range(count):
            sequence1, newx4 = a_1.random_generate()
            sequence2 = [i+1 for i in sequence1]

            _, newx40 = a_2.random_generate()
            sequence0 = [i-1 for i in sequence1]
            f = a_0.offline(sequence0)  # optimal result
            optimal = np.dot(multi, f)
            accept_people += optimal
            
            g0 = a_2.method_new(sequence2, newx40, roll_width+1)
            sto_2 += np.dot(multi, g0)

            g = a_1.method_new(sequence1, newx4, roll_width)
            sto_1 += np.dot(multi, g)
            

        occup_1[cnt] = sto_1/count/total_seat * 100
        occup_2[cnt] = sto_2/count/total_seat * 100
        people_value[cnt] = accept_people/count/total_seat * 100

        cnt += 1
    plt.plot(t_value, people_value, 'b-', label = 'With 0 social distancing')
    plt.plot(t_value, occup_1, 'r--', label = 'With 1 social distancing')
    plt.plot(t_value, occup_2, 'y*', label='With 2 social distancing')

    # plt.plot(t_value, occup_without, 'y*', label= 'Dynamic Without SD')
    plt.xlabel('Periods')
    plt.ylabel('Percentage of total seats')

    plt.legend()
    plt.show()
