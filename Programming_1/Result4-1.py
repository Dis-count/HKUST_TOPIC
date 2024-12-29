import numpy as np
from Comparison import CompareMethods
import matplotlib.pyplot as plt
import copy
import time

# Find the performance of dp any probability combinations

def prop_all():
    x1 = np.arange(0.05, 1, 0.05)
    x2 = np.arange(0.05, 1, 0.05)
    x3 = np.arange(0.05, 1, 0.05)

    p = np.zeros((len(x1)*len(x2)*len(x3), 4))
    t = 0

    for i in x1:
        for j in x2:
            for k in x3:
                if 1 - i - j - k > 0:
                    p[t] = [i, j, k, 1 - i - j - k]
                    t += 1

    p = p[0:t]

    return p

# gamma = 2.5
def prop_list():
    x = np.arange(0.05, 1, 0.05)
    y = np.arange(0.05, 0.8, 0.05)
    p = np.zeros((len(x)*len(y), 4))

    t = 0
    for i in x:
        for j in y:
            if 3-2*i-4*j > 0 and 3-4*i-2*j > 0:
                p[t] = [(3 - 4*i - 2*j)/6, i, j, (3 - 2*i - 4*j)/6]
                t += 1
    p = p[0:t]

    return p

# gamma = 2
def prop_list1():
    x = np.arange(0.05, 0.5, 0.1)  # p3
    y = np.arange(0.05, 0.35, 0.05)  # p4
    p = np.zeros((len(x)*len(y), 4))

    t = 0
    for i in x:
        for j in y:
            if 1-2*i-3*j > 0:
                p[t] = [(i + 2*j), (1 - 2*i - 3*j), i, j]
                t += 1
    p = p[0:t]

    return p

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(70,71,1)
    given_lines = 10
    # probab = [0.3, 0.2, 0.2, 0.3]
    p = prop_list()
    s = 1
    # p = [[0.3, 0.2, 0.2, 0.3]]
    begin_time = time.time()
    filename = 'Periods_' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')


    for probab in p:
        my_file.write(str(probab))
        gap_if = True
        cnt = 100
        for num_period in period_range:
            roll_width = np.ones(given_lines) * 21
            ratio = 0
            for j in range(cnt):
                a = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, s)
                sequence, newx4 = a.random_generate()

                dp = a.dynamic_program1(sequence)
                multi = np.arange(1, I+1)

                dp_value = np.dot(multi, dp)

                f = a.offline(sequence)
                optimal = np.dot(multi, f)
                ratio += dp_value / optimal

            my_file.write('DP1: %.2f ;\n' % (ratio/cnt*100))

    my_file.close()
