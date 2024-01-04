import numpy as np
from Comparison import CompareMethods
import time

def prop_list():
    x = np.arange(0.05, 1, 0.05)
    y = np.arange(0.05, 0.8, 0.05)
    p = np.zeros((len(x)*len(y), 4))

    t = 0
    for i in x:
        for j in y:
            if 3-2*i-4*j > 0 and 3- 4*i- 2*j > 0:
                p[t] = [(3 - 4*i - 2*j)/6, i, j, (3 - 2*i - 4*j)/6]
                t += 1
    p = p[0:t]
    return p

# Results of Different Policies under one period and one probability
if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 70
    given_lines = 10
    sd = 1
    prop = prop_list()
    roll_width = np.ones(given_lines) * 21

    begin_time = time.time()
    filename = 'Periods_' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    for p in prop:
        my_file.write('Probability: \t' + str(p) + '\n')
        print(p)
        a_instance = CompareMethods(roll_width, given_lines, I, p, num_period, num_sample, sd)

        ratio1 = 0
        ratio2 = 0
        ratio3 = 0
        ratio4 = 0
        ratio5 = 0
        accept_people = 0

        multi = np.arange(1, I+1)
        count = 1

        for j in range(count):
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()

            a = a_instance.method_new(sequence, newx4, roll_width)
            b = a_instance.bid_price(sequence)
            c = a_instance.dynamic_program1(sequence)
            d = a_instance.method_IP(sequence, newx3, roll_width)
            e = a_instance.row_by_row(sequence)

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            ratio1 += np.dot(multi, a) / optimal   # sto-planning
            ratio2 += np.dot(multi, b) / optimal   # bid-price
            ratio3 += np.dot(multi, c) / optimal   # DP1
            ratio4 += np.dot(multi, d) / optimal   # booking
            ratio5 += np.dot(multi, e) / optimal   # FCFS
            accept_people += optimal

        my_file.write('Sto: %.2f ;' % (ratio1/count*100))
        my_file.write('Bid: %.2f ;' % (ratio2/count*100))
        my_file.write('DP1: %.2f ;' % (ratio3/count*100))
        my_file.write('Booking: %.2f ;' % (ratio4/count*100))
        my_file.write('FCFS: %.2f ;' % (ratio5/count*100))
        my_file.write('Number of accepted people: %.2f \n' % (accept_people/count))

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t%f\n' % run_time)

    my_file.close()
