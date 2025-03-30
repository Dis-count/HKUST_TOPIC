import numpy as np
from Comparison import CompareMethods
import time
from Mist import sequence_pool
from Method10 import deterministicModel
# Results of Different Policies under multiple probabilities

# [0.18, 0.7, 0.06, 0.06], [0, 0.5, 0, 0.5],
# [0.2, 0.8, 0, 0], [0, 1, 0, 0]
# [0.05, 0.05, 0.85, 0.05], [0.25, 0.3, 0.25, 0.2]

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(60, 101, 10)
    given_lines = 10
    # probab_list = [[0.2, 0.8, 0, 0], [0.18, 0.7, 0.06, 0.06], [0.12, 0.5, 0.13, 0.25], [0.34, 0.51, 0.07, 0.08]]
    probab_list = [[0.18, 0.7, 0.06, 0.06]]

    sd = 1
    count = 100
    total_period = 100

    for probab in probab_list:
        begin_time = time.time()
        filename = '330_probab_' + str(probab) + '.txt'
        my_file = open(filename, 'w')
        my_file.write('Run Start Time:' + str(time.ctime()) + '\n')
        sequences_pool = np.load('sequence_0.18.npy')
        # sequences_pool = sequence_pool(count, total_period, probab, sd)

        for num_period in period_range:
            worst_a = 1
            worst_b = 1
            worst_c = 1
            worst_d = 1
            my_file.write('The number of periods: \t' + str(num_period) + '\n')
            
            roll_width = np.ones(given_lines) * 21
            a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)

            ratio1 = 0
            ratio2 = 0
            ratio3 = 0
            ratio4 = 0
            accept_people = 0

            multi = np.arange(1, I+1)

            # Generate booking-limit
            demand_width_array = np.arange(1+sd, 1+sd+I)
            deter = deterministicModel(roll_width, given_lines, demand_width_array, I, sd)

            ini_demand1 = np.array(probab) * num_period
            ini_demand3, _ = deter.IP_formulation(np.zeros(I), ini_demand1)
            _, newx3 = deter.IP_formulation(ini_demand3, np.zeros(I))
            
            for j in range(count):
                sequence = sequences_pool[j][0: num_period]
                newx4 = a_instance.random_generate(sequence)

                a = a_instance.method_new(sequence, newx4, roll_width)
                b = a_instance.bid_price(sequence)
                c = a_instance.dynamic_program1(sequence)
                d = a_instance.method_IP(sequence, newx3, roll_width)
                # e = a_instance.row_by_row(sequence)
                value_a = np.dot(multi, a)
                value_b = np.dot(multi, b)
                value_c = np.dot(multi, c)
                value_d = np.dot(multi, d)

                f = a_instance.offline(sequence)  # optimal result
                optimal = np.dot(multi, f)

                if value_a/optimal < worst_a:
                    worst_a = value_a/optimal

                if value_b/optimal < worst_b:
                    worst_b = value_b/optimal

                if value_c/optimal < worst_c:
                    worst_c = value_c/optimal

                if value_d/optimal < worst_d:
                    worst_d = value_d/optimal

                ratio1 += value_a / optimal  # sto-planning
                ratio2 += value_b / optimal  # bid-price
                ratio3 += value_c / optimal  # DP1
                ratio4 += value_d / optimal  # booking-limit
                # ratio5 += np.dot(multi, e) / optimal  # FCFS
                accept_people += optimal

            my_file.write('SPBA: %.2f ;' % (ratio1/count*100))
            my_file.write('RDPH: %.2f ;' % (ratio3/count*100))
            my_file.write('BPC: %.2f ;' % (ratio2/count*100))
            my_file.write('BLC: %.2f ;' % (ratio4/count*100))
            my_file.write('Number of accepted people: %.2f \t' % (accept_people/count))
            my_file.write('worst_SPBA: %.4f \n;' % (worst_a))
            my_file.write('worst_BPC: %.4f \n;' % (worst_b))
            my_file.write('worst_RDPH: %.4f \n;' % (worst_c))
            my_file.write('worst_BLC: %.4f \n;' % (worst_d))

        run_time = time.time() - begin_time
        my_file.write('Total Runtime\t%f\n' % run_time)
            # print('%.2f' % (ratio1/count*100))
        my_file.close()