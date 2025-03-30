import numpy as np
from Comparison import CompareMethods
from Mist import sequence_pool

# This function is used to record all the data from the simulations.

# The specific parameters are as follows:
# The occupancy rate with different group sizes 
# [0.2, 0.2, 0.6]  3
# [0.25, 0.3, 0.25, 0.2]  4

# output:
# data_group4.npy
# data_group3.npy
# data_group2.npy

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    # I = 5  # the number of group types
    total_period = 100
    period_range = range(40, total_period, 1)
    given_lines = 10
    sd = 1
    # 2: [0.19, 0.81],   [0.4, 0.6]
    # 3: [0.16, 0.67, 0.17],  [0.37, 0.55, 0.08]
    # 4: [0.12, 0.5, 0.13, 0.25]   [0.34, 0.51, 0.07, 0.08]
    group_dict = {3: [0.16, 0.67, 0.17],
                  2: [0.19, 0.81]}

    t_value = np.arange(40, total_period, 1)
    people_value = np.zeros(len(period_range))
    occup_value = np.zeros(len(period_range))
    count = 100

    for I, probab in group_dict.items():
        sequences_pool = sequence_pool(count, total_period, probab, sd)
        # sequences_pool = np.load('sequence_0.12.npy')
        cnt = 0
        gap_if = True
        
        for num_period in period_range:
            roll_width = np.ones(given_lines) * 21
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
                # newx40 = a_without.random_generate(sequence1)
                if num_period <= 70:
                    f = a_without.offline(sequence1)  # optimal result
                else:
                    f = a_without.method_new(sequence1, newx40, roll_width-sd)
                accept_people += np.dot(multi, f)

                g = a_instance.method_new(sequence, newx4, roll_width)
                sto += np.dot(multi, g)
                
            occup_value[cnt] = sto/count/total_seat * 100
            people_value[cnt] = accept_people/count/total_seat * 100
            if gap_if:
                if accept_people/count - sto/count > 1:
                    point = [num_period-1, occup_value[cnt-1]]
                    gap_if = False
            cnt += 1

        data = np.vstack((t_value, people_value, occup_value))
        np.save('d3_group_' + str(I) + '.npy', data)
