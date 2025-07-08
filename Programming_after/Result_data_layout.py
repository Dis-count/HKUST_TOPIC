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
    total_period = 50
    period_range = range(20, total_period, 1)
    # given_lines = 11

    # layout_dic = {'fan': np.array([17, 18, 19, 20, 21, 21, 22, 23, 24, 25]),
    #               'rec_20rows': np.ones(20) * 11,
    #               'small': np.ones(15) * 8,
    #               'rec_10rows': np.ones(10) * 21}

    # layout_dic = {'small': np.array([5, 7, 9, 11, 11]),
    #   'large': np.ones(11) * 15, 'small': np.ones(6) * 7, 'large_not_rec': np.array([15, 15, 15, 15, 15, 15, 17, 17, 17, 17, 17, 17, 17, 17, 17, 19])}
    layout_dic = {'hk_fac': np.array([17, 18, 18, 18, 18, 18, 18, 8])}

    probab = [0.25, 0.3, 0.25, 0.2]
    sd = 1
    t_value = np.arange(20, total_period, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))
    count = 100
    sequences_pool = np.load('data_sequence0.25.npy')

    for shape, roll_width in layout_dic.items():
        cnt = 0
        given_lines = len(roll_width)

        for num_period in period_range:
            total_seat = np.sum(roll_width) - given_lines * sd
            a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)
            a_without = CompareMethods(roll_width-sd, given_lines, I, probab, num_period, num_sample, 0)
            sto = 0
            accept_people = 0
            multi = np.arange(1, I+1)

            for j in range(count):
                sequence = sequences_pool[j][0: num_period]
                newx4 = a_instance.random_generate(sequence)
                sequence1 = [i-sd for i in sequence]
                newx40 = a_without.random_generate(sequence1)

                f = a_without.offline(sequence1)  # optimal result
                accept_people += np.dot(multi, f)

                g = a_instance.method_new(sequence, newx4, roll_width)
                sto += np.dot(multi, g)

            occup_value[cnt] = sto/count/total_seat * 100
            people_value[cnt] = accept_people/count/total_seat * 100

            cnt += 1

        data = np.vstack((t_value, people_value, occup_value))
        np.save('layout_' + str(shape) + '.npy', data)