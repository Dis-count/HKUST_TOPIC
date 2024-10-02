#  This file is used to add two more programmings
#  1. location of rejecting 1 group

import numpy as np
from Comparison import CompareMethods
import time
from collections import Counter

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 70
    given_lines = 10
    s = 1
    p = [0.25, 0.25, 0.25, 0.25]
    begin_time = time.time()

    roll_width = np.ones(given_lines) * 21
    a_instance = CompareMethods(roll_width, given_lines, I, p, num_period, num_sample, s)

    ratio1 = 0
    accept_people = 0

    multi = np.arange(1, I+1)
    count = 1

    for j in range(count):
        sequence, ini_demand, newx4 = a_instance.random_generate()
        
        a, mylist = a_instance.method_new_copy(sequence, newx4, roll_width)

        f = a_instance.offline(sequence)  # optimal result
        optimal = np.dot(multi, f)

        opt_list = []
        for i in sequence:
            if f[i-1-s] > 0:
                opt_list.append(1)
                f[i-1-s] -= 1
            else:
                opt_list.append(0)
        opt_list = np.array(opt_list)
        bb = np.where(opt_list == 0)[0]
        reject_opt_list = [sequence[i] for i in bb]
        t2 = Counter(reject_opt_list)
        print(t2)

        mylist = np.array(mylist)  # 找到拒绝的位置
        aa = np.where(mylist == 0)[0]

        reject_list = [sequence[i] for i in aa]

        t = Counter(reject_list)
        print(t)