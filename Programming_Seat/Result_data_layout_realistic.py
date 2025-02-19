import numpy as np
from Comparison import CompareMethods
from Mist import sequence_pool
# This function is used to record all the data from the simulations.
# The specific parameters are as follows:

# The layouts are:
# [0.25, 0.3, 0.25, 0.2]  I = 4  gamma = 2.4

# np.array([17, 18, 18, 18, 18, 18, 18, 8]) given_lines = 8  
# length = 133  gap = 39

# np.array([7, 9, 7, 10, 7, 11, 8, 11, 9, 7, 10, 7, 11, 7, 13, 8])  given_lines = 16
# length = 142  gap = 42

# np.array([3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])  given_lines = 13
#  length = 159  gap = 47

# np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])  given_lines = 22
#  length = 132  gap = 39

# np.array([13, 21, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 21, 13]) 
# given_lines = 17, length = 367  gap = 108

# output: 


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    total_period = 110
    period_range = range(90, total_period, 1)
    # given_lines = 11

    # layout_dic = {'HK_FAC': np.array([17, 18, 18, 18, 18, 18, 18, 8]),
                #   'KTT_TS': np.array([7, 9, 7, 10, 7, 11, 8, 11, 9, 7, 10, 7, 11, 7, 13, 8])}
    layout_dic = {'NCWCC': np.array([13, 21, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 21, 13])}

    # layout_dic = {'HK_FAC': np.array([17, 18, 18, 18, 18, 18, 18, 8]),
    #               'KTT_TS': np.array([7, 9, 7, 10, 7, 11, 8, 11, 9, 7, 10, 7, 11, 7, 13, 8]),
    #               'SWHCC': np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
    #               'SWCC': np.array([3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])}

    probab = [0.12, 0.5, 0.13, 0.25]
    sd = 1
    t_value = np.arange(90, total_period, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))
    count = 100
    # sequences_pool = np.load('sequence_0.12.npy')
    sequences_pool = sequence_pool(count, total_period, probab, sd)

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