import numpy as np
from Comparison import CompareMethods
from Mist import sequence_pool

# This function is used to record all the data from the simulations.
# The specific parameters are as follows:

# The social distancing is 0,1,2 respectively.
# [0.25, 0.3, 0.25, 0.2]

# print out:
# data_distances

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    total_period = 100
    period_range = range(40, total_period, 1)
    given_lines = 10
    sd = 1
    # probab = [0.12, 0.5, 0.13, 0.25]
    # probab = [0.16, 0.67, 0.17]
    probab = [0.19, 0.81]

    I = len(probab)

    t_value = np.arange(40, total_period, 1)

    people_value = np.zeros(len(period_range))
    occup_1 = np.zeros(len(period_range))
    occup_2 = np.zeros(len(period_range))
    count = 100
    # sequences_pool = sequence_pool(count, total_period, probab, sd)
    sequences_pool = np.load('sequence_M2_0.19.npy')
    
    cnt = 0
    for num_period in period_range:

        roll_width = np.ones(given_lines) * 21
        total_seat = np.sum(roll_width) - given_lines * sd

        a_0 = CompareMethods(roll_width - 1, given_lines, I, probab, num_period, num_sample, 0)
        a_1 = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)
        a_2= CompareMethods(roll_width + 1, given_lines, I, probab, num_period, num_sample, sd+1)

        sto_1 = 0
        sto_2 = 0
        accept_people = 0
        multi = np.arange(1, I+1)

        for j in range(count):
            sequence1 = sequences_pool[j][0: num_period]
            newx4 = a_1.random_generate(sequence1)

            sequence2 = [i+1 for i in sequence1]
            newx40 = a_2.random_generate(sequence2)

            sequence0 = [i-1 for i in sequence1]
            f = a_0.offline(sequence0)  # optimal result
            accept_people += np.dot(multi, f)
            
            g0 = a_2.method_new(sequence2, newx40, roll_width+1)
            sto_2 += np.dot(multi, g0)

            g = a_1.method_new(sequence1, newx4, roll_width)
            sto_1 += np.dot(multi, g)

        occup_1[cnt] = sto_1/count/total_seat * 100
        occup_2[cnt] = sto_2/count/total_seat * 100
        people_value[cnt] = accept_people/count/total_seat * 100
        cnt += 1

    data = np.vstack((t_value, people_value, occup_1, occup_2))
    np.save('data_distances_012_M3.npy', data)
