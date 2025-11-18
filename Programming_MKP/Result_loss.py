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
    period_range = range(300, 301, 10)
    given_lines = 10
    # probab_list = [[0.2, 0.5, 0.3]]
    probab_list = [[0.25, 0.25, 0.25, 0.25]]
    # value = np.array([5, 4, 3])
    value = np.array([2, 4, 6, 8])
    weight = np.array([3, 5, 7, 9])
    # value = weight
    # weight = np.array([8, 6, 4])
    count = 100
    total_period = 300

    for probab in probab_list:
        begin_time = time.time()
        filename = 'loss_m=4_300' + str(probab) + '.txt'
        my_file = open(filename, 'w')
        my_file.write('Run Start Time:' + str(time.ctime()) + '\n')
        sequences_pool = sequence_pool(count, total_period, probab)

        for num_period in period_range:
            my_file.write('The number of periods: \t' + str(num_period) + '\n')
            
            roll_width = np.ones(given_lines) * 20 * 5
            a_instance = CompareMethods(roll_width, given_lines, I, num_period, value, weight, probab)

            loss1 = 0
            loss2 = 0
            loss3 = 0
            revenue = 0

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

                loss1 += optimal-value_a  # Primal
                loss2 += optimal-value_b  # BPC
                loss3 += optimal-value_c  # BPP
                revenue += optimal

            my_file.write('Primal: %.2f ;' % (loss1/count))
            my_file.write('BPC: %.2f ;' % (loss2/count))
            my_file.write('BPP: %.2f ;' % (loss3/count))
            my_file.write('Average Optimal Revenue: %.2f \n' % (revenue/count))
        

        run_time = time.time() - begin_time
        my_file.write('Total Runtime\t%f\n' % run_time)
            # print('%.2f' % (ratio1/count*100))
        my_file.close()