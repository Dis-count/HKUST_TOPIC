# import gurobipy as grb
# from gurobipy import GRB
import numpy as np
from Comparison_new import CompareMethod_new
import time

# Results of Different Policies under multiple periods and one probability
if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(120,121,10)
    given_lines = 10
    probab = [0.25, 0.25, 0.25, 0.25]
    s = 1
    begin_time = time.time()
    filename = 'Periods_' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    for num_period in period_range:
        my_file.write('The number of periods: \t' + str(num_period) + '\n')
        
        roll_width = np.ones(given_lines) * 21
        # roll_width = np.array([17,18,19,20,21,21,22,23,24,20])
        a_instance = CompareMethod_new(roll_width, given_lines, I, probab, num_period, num_sample, s)

        ratio1 = 0
        ratio2 = 0
        ratio3 = 0
        ratio4 = 0
        ratio5 = 0
        accept_people =0

        multi = np.arange(1, I+1)
        count = 100

        value = a_instance.dynamic2(220, 220, 151)
        aaa = np.array(value)
        np.save('a_1.npy', aaa)

        aaa = np.load('a_1.npy')
        value = aaa.tolist()

        for j in range(count):
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()

            a = a_instance.method_new(sequence, newx4, roll_width)
            b = a_instance.main_dy(sequence, value)
            # c = a_instance.dynamic_program1(sequence)
            # d = a_instance.method_IP(sequence, newx3, roll_width)
            # e = a_instance.row_by_row(sequence)

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            ratio1 += np.dot(multi, a) / optimal  # sto-planning
            ratio2 += np.dot(multi, b) / optimal  # bid-price
            # ratio3 += np.dot(multi, c) / optimal  # DP1
            # ratio4 += np.dot(multi, d) / optimal  # booking-limit
            # ratio5 += np.dot(multi, e) / optimal  # FCFS
            accept_people += optimal

        my_file.write('Sto: %.2f ;' % (ratio1/count*100))
        my_file.write('DP_relaxed: %.2f ;' % (ratio2/count*100))
        # my_file.write('DP1: %.2f ;' % (ratio3/count*100))
        # my_file.write('Booking: %.2f ;' % (ratio4/count*100))
        # my_file.write('FCFS: %.2f \n;' % (ratio5/count*100))
        my_file.write('Number of accepted people: %.2f \t' % (accept_people/count))

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t%f\n' % run_time)
        # print('%.2f' % (ratio1/count*100))

    my_file.close()