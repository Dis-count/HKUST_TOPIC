import numpy as np
from I_BPC import CompareMethods
import time
from Mist import sequence_pool
from Method10 import deterministicModel
# Results of Different Policies under multiple probabilities

# [0.18, 0.7, 0.06, 0.06], [0, 0.5, 0, 0.5],
# [0.2, 0.8, 0, 0], [0, 1, 0, 0]
# [0.05, 0.05, 0.85, 0.05], [0.25, 0.3, 0.25, 0.2]

if __name__ == "__main__":
    I = 4  # the number of group types
    period_range = range(60, 61, 10)
    given_lines = 10
    # probab_list = [[0.2, 0.5, 0.3]]
    probab_list = [[0.1, 0.1, 0.25, 0.25, 0.15, 0.15]]
    # probab_list = [[0.25, 0.25, 0.25, 0.25]]
    value = np.array([5, 5, 4, 4, 3, 3])
    # value = np.array([2, 4, 6, 8])
    # weight = np.array([3, 5, 7, 9])
    # value = weight
    weight = np.array([8, 8, 6, 6, 4, 4])
    count = 100
    total_period = 100

    for probab in probab_list:
        begin_time = time.time()
        filename = 'devide_m=4_60' + str(probab) + '.txt'
        my_file = open(filename, 'w')
        my_file.write('Run Start Time:' + str(time.ctime()) + '\n')
        # sequences_pool = np.load('sequence_0.12.npy')
        sequences_pool = sequence_pool(count, total_period, probab)

        for num_period in period_range:
            my_file.write('The number of periods: \t' + str(num_period) + '\n')
            
            roll_width = np.ones(given_lines) * 20
            a_instance = CompareMethods(roll_width, given_lines, I, num_period, value, weight, probab)

            ratio1 = 0
            ratio2 = 0
            ratio3 = 0
            accept_people = 0

            for j in range(count):
                sequence = sequences_pool[j][0: num_period]
                a = a_instance.dynamic_primal(sequence)
                b = a_instance.bid_price_1(sequence)
                c = a_instance.improved_bid(sequence)

                value_a = np.dot(value, a)
                value_b = np.dot(value, b)
                value_c = np.dot(value, c)

                f = a_instance.offline(sequence)  # optimal result
                optimal = np.dot(value, f)

                ratio1 += value_a / optimal  # Primal
                ratio2 += value_b / optimal  # BPC
                ratio3 += value_c / optimal  # BPP
                accept_people += optimal

            my_file.write('Primal: %.2f ;' % (ratio1/count*100))
            my_file.write('BPC: %.2f ;' % (ratio2/count*100))
            my_file.write('BPP: %.2f ;' % (ratio3/count*100))
            my_file.write('Average Optimal Revenue: %.2f \n' % (accept_people/count))
        

        run_time = time.time() - begin_time
        my_file.write('Total Runtime\t%f\n' % run_time)
            # print('%.2f' % (ratio1/count*100))
        my_file.close()