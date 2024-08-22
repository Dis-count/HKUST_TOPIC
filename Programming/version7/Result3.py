import numpy as np
from Comparison import CompareMethods
import time

# Results of Different Policies under one period and one probability
if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 80
    given_lines = 10

    p = [0.25, 0.25, 0.25, 0.25]
    begin_time = time.time()

    roll_width = np.ones(given_lines) * 21

    a_instance = CompareMethods(roll_width, given_lines, I, p, num_period, num_sample)

    ratio1 = 0
    ratio2 = 0
    ratio3 = 0
    ratio4 = 0
    ratio5 = 0
    accept_people = 0

    multi = np.arange(1, I+1)
    count = 100

    for j in range(count):
        sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()

        a = a_instance.method_new(sequence, newx4, roll_width)
        b = a_instance.bid_price(sequence)
        c = a_instance.dynamic_program1(sequence)
        d = a_instance.method1(sequence, ini_demand)
        e = a_instance.dynamic_program(sequence)

        f = a_instance.offline(sequence)  # optimal result
        optimal = np.dot(multi, f)

        ratio1 += np.dot(multi, a) / optimal   # sto-planning
        ratio2 += np.dot(multi, b) / optimal   # bid-price
        ratio3 += np.dot(multi, c) / optimal   # DP1
        ratio4 += np.dot(multi, d) / optimal   # once
        ratio5 += np.dot(multi, e) / optimal   # DP-based
        accept_people += optimal

    print('%.2f' % (ratio1/count*100))
    print('%.2f' % (ratio2/count*100))
    print('%.2f' % (ratio3/count*100))
    print('%.2f' % (ratio4/count*100))
    print('%.2f' % (ratio5/count*100))
