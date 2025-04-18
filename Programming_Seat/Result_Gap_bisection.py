import numpy as np
from Comparison import CompareMethods
from Mist import sequence_pool
import time
import random

# Find the gap point by bisection

def prop_all():
    x2 = np.arange(0.05, 1, 0.05)
    x3 = np.arange(0.05, 1, 0.05)
    x4 = np.arange(0.05, 1, 0.05)

    p = np.zeros((len(x2)*len(x3)*len(x4), 4))
    t = 0

    for i in x2:
        for j in x3:
            for k in x4:
                if 1 - i - j - k > 0:
                    p[t] = [1 - i - j - k, i, j, k]
                    t += 1

    p = p[0:t]

    return p

# # gamma = 2.5
# def prop_list():
#     x = np.arange(0.05, 1, 0.05)
#     y = np.arange(0.05, 0.8, 0.05)
#     p = np.zeros((len(x)*len(y), 4))

#     t = 0
#     for i in x:
#         for j in y:
#             if 3-2*i-4*j > 0 and 3-4*i-2*j > 0:
#                 p[t] = [(3 - 4*i - 2*j)/6, i, j, (3 - 2*i - 4*j)/6]
#                 t += 1
#     p = p[0:t]

#     return p

# # gamma = 2
# def prop_list1():
#     x = np.arange(0.05, 0.5, 0.1)  # p3
#     y = np.arange(0.05, 0.35, 0.05)  # p4
#     p = np.zeros((len(x)*len(y), 4))

#     t = 0
#     for i in x:
#         for j in y:
#             if 1-2*i-3*j > 0:
#                 p[t] = [(i + 2*j), (1 - 2*i - 3*j), i, j]
#                 t += 1
#     p = p[0:t]

#     return p

# def prop_list_19():
#     x = np.arange(0.05, 0.9, 0.05)  # p2
#     y = np.arange(0.05, 0.45, 0.05)  # p3
#     p = np.zeros((len(x)*len(y), 4))

#     t = 0
#     for i in x:
#         for j in y:
#             if 0.7 - 2*i/3 - j/3 > 0 and 0.3 - i/3 - 2*j/3 > 0:
#                 p[t] = [(0.7 - 2*i/3 - j/3), i, j, (0.3 - i/3 - 2*j/3)]
#                 t += 1
#     p = p[0:t]

#     return p

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    given_lines = 20
    probab = prop_all()
    # 随机100个数字
    probab_list = np.load('Prob_random.npy')
    total_period = 100
    sd = 1
    begin_time = time.time()

    filename = 'Random_100_20rows.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')
    count = 100

    for i in range(100):
        p = probab[probab_list[i]]
        sequences_pool = sequence_pool(count, total_period, p, sd)
        my_file.write(str(p) + '\t')

        a, b = 30, 100
        num_period = int((a+b)/2)
        while num_period > a:
            roll_width = np.ones(given_lines) * 11
            # roll_width = np.array([17, 18, 19, 20, 21, 21, 22, 23, 24, 25])
            total_seat = np.sum(roll_width) - given_lines
            a_instance = CompareMethods(roll_width, given_lines, I, p, num_period, num_sample, sd)
            a_without = CompareMethods(roll_width-sd, given_lines, I, p, num_period, num_sample, 0)
            sto = 0
            accept_people = 0
            multi = np.arange(1, I+1)
            
            for j in range(count):
                sequence = sequences_pool[j][0: num_period]
                newx4 = a_instance.random_generate(sequence)

                sequence1 = [i-sd for i in sequence]
                f = a_without.offline(sequence1)  # optimal result
                accept_people += np.dot(multi, f)
                
                g = a_instance.method_new(sequence, newx4, roll_width)
                sto += np.dot(multi, g)

            occup_value = sto/count/total_seat * 100

            if accept_people/count - sto/count > 1:
                b = num_period
                num_period = int((a + b)/2)
            else:
                a = num_period
                num_period = int((a + b)/2)

        point = [num_period, occup_value]
        my_file.write(str(point) + '\n')

    my_file.close()