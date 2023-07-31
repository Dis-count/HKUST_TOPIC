# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from Comparison import CompareMethods
import time

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(50,101,10)
    given_lines = 10
    # np.random.seed(i)
    probab = [0.25, 0.25, 0.25, 0.25]

    begin_time = time.time()
    filename = 'Periods_' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    for num_period in period_range:
        my_file.write('The number of periods: \t' + str(num_period) + '\n')
        
        roll_width = np.ones(given_lines) * 21
        a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)

        ratio1 = 0
        ratio2 = 0
        ratio3 = 0
        ratio4 = 0
        accept_people =0
        num_people = 0

        multi = np.arange(1, I+1)

        count = 100
        for j in range(count):
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()

            total_people = sum(sequence) - num_period

            a = a_instance.method_new(sequence, ini_demand, newx4, roll_width)
            b = a_instance.bid_price(sequence)

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            d = a_instance.dynamic_program1(sequence)
            e = a_instance.row_by_row(sequence)

            ratio1 += np.dot(multi, a) / optimal
            ratio2 += np.dot(multi, b) / optimal
            ratio3 += np.dot(multi, d) / optimal
            ratio4 += np.dot(multi, e) / optimal
            accept_people += optimal
            num_people += total_people

        my_file.write('Sto: %.2f ;' % (ratio1/count*100))
        my_file.write('Bid: %.2f ;' % (ratio2/count*100))
        my_file.write('DP1: %.2f ;' % (ratio3/count*100))
        my_file.write('FCFS: %.2f \n;' % (ratio4/count*100))
        my_file.write('Number of accepted people: %.2f \t' % (accept_people/count))
        my_file.write('Number of people: %.2f \n' % (num_people/count))

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t%f\n' % run_time)
        # print('%.2f' % (ratio1/count*100))

    my_file.close()