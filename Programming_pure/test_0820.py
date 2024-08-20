#  This file is used to add two more programmings
#  1. location of rejecting 1 group
# 
#  2. compare occupancy rate in optimal accepted individuals versus seat planning under 200 instances

import numpy as np
from Comparison import CompareMethods
import time


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    num_period = 80
    given_lines = 10
    s = 1
    p = [0.25, 0.25, 0.25, 0.25]
    begin_time = time.time()

    roll_width = np.ones(given_lines) * 21

    a_instance = CompareMethods(roll_width, given_lines, I, p, num_period, num_sample, s)

    ratio1 = 0
    accept_people = 0

    multi = np.arange(1, I+1)
    count = 100
    lower_plan = 0

    for j in range(count):
        sequence, ini_demand, newx4 = a_instance.random_generate()
        
        seat_plan = np.dot(multi, ini_demand)

        a = a_instance.method_new(sequence, newx4, roll_width)

        e = a_instance.dynamic_program(sequence)

        f = a_instance.offline(sequence)  # optimal result
        optimal = np.dot(multi, f)

        ratio1 += np.dot(multi, a) / optimal   # sto-planning

        accept_people += optimal

        lower_plan += (seat_plan < optimal)

    print(lower_plan/count)

        