import numpy as np
from Comparison import CompareMethods

# This function is used to record all the data from the simulations.
# The specific parameters are as follows:

# The layouts are:
# [0.25, 0.3, 0.25, 0.2] I = 4
# np.array([17, 18, 19, 20, 21, 21, 22, 23, 24, 25]) given_lines = 10
# np.ones(given_lines) * 11  given_lines = 20
# np.ones(given_lines) * 21  given_lines = 10
# np.ones(given_lines) * 8  given_lines = 15

# output: 
# data_layout_10: fan-shaped 
# data_layout_20: 20 rows
# data_layout_15: 15 rows small room

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(30, 100, 1)
    # given_lines = 11
    layout_dic = {10: np.array([17, 18, 19, 20, 21, 21, 22, 23, 24, 25]),
                  20: np.ones(20) * 11,
                  15: np.ones(15) * 8}

    probab = [0.25, 0.3, 0.25, 0.2]
    sd = 1
    t_value = np.arange(30, 100, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))
    cnt = 0
    gap_if = True

    for given_lines, roll_width in layout_dic.items():
        for num_period in period_range:
            total_seat = np.sum(roll_width) - given_lines * sd
            a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)
            a_without = CompareMethods(roll_width-sd, given_lines, I, probab, num_period, num_sample, 0)
            sto = 0
            accept_people = 0
            multi = np.arange(1, I+1)

            count = 100
            for j in range(count):
                sequence, newx4 = a_instance.random_generate()
                sequence1 = [i-sd for i in sequence]
                _, newx40 = a_without.random_generate()
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
                    gap_if = False
            cnt += 1

        data = np.vstack((t_value, people_value, occup_value))

        np.save('data_layout' + str(given_lines) + '.npy', data)

