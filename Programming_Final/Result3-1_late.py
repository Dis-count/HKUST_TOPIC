# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from Comparison_late import CompareMethods
import time

# performance of the later seat assignment
# Results of Different Policies under multiple periods and one probability

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(60, 101, 10)
    given_lines = 10
    probab = [0.25, 0.35, 0.05, 0.35]
    s = 1
    begin_time = time.time()
    filename = 'Periods_' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    for num_period in period_range:
        my_file.write('The number of periods: \t' + str(num_period) + '\n')
        
        roll_width = np.ones(given_lines) * 21
        a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, s)

        ratio2 = 0
        ratio3 = 0
        accept_people =0

        multi = np.arange(1, I+1)
        count = 100

        for j in range(count):
            sequence, ini_demand, newx4 = a_instance.random_generate()

            b = a_instance.bid_price(sequence)
            c = a_instance.dynamic_program(sequence)

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            ratio2 += np.dot(multi, b) / optimal  # bid-price
            ratio3 += np.dot(multi, c) / optimal  # DP1

            accept_people += optimal

        my_file.write('Bid: %.2f ;' % (ratio2/count*100))
        my_file.write('DP1: %.2f ;' % (ratio3/count*100))

        my_file.write('Number of accepted people: %.2f \t' % (accept_people/count))

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t%f\n' % run_time)
        # print('%.2f' % (ratio1/count*100))

    my_file.close()