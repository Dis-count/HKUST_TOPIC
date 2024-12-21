import numpy as np
from Comparison import CompareMethods
import time
from Mist import sequence_pool
from Method10 import deterministicModel
# Results of Different Policies under multiple probabilities

# [0.18, 0.7, 0.06, 0.06], [0, 0.5, 0, 0.5]

# [0.2, 0.8, 0, 0], [0, 1, 0, 0]

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(60, 201, 10)
    given_lines = 10
    probab = [0.18, 0.7, 0.06, 0.06]
    total_period = 200
    sd = 1
    count = 1

    sequences_pool = np.load('data_sequence0.18.npy')
    # sequences_pool = sequence_pool(count, total_period, probab, sd)
    num_period = 90
 
    roll_width = np.ones(given_lines) * 21
    a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)

    ratio1 = 0
    ratio2 = 0
    ratio3 = 0
    ratio4 = 0
    ratio5 = 0
    accept_people = 0

    multi = np.arange(1, I+1)

    # Generate booking-limit
    demand_width_array = np.arange(1+sd, 1+sd+I)
    deter = deterministicModel(roll_width, given_lines, demand_width_array, I, sd)

    ini_demand1 = np.array(probab) * num_period
    ini_demand3, _ = deter.IP_formulation(np.zeros(I), ini_demand1)
    _, newx3 = deter.IP_formulation(ini_demand3, np.zeros(I))

    sequence = sequences_pool[3][0: num_period]
    print(sequence)

    total_demand = np.zeros(I)
    for i in sequence:
        total_demand[i-1-sd] += 1
    print(total_demand)

    newx4 = a_instance.random_generate(sequence)
    a = a_instance.method_new(sequence, newx4, roll_width)
    b = a_instance.bid_price(sequence)
    c = a_instance.dynamic_program1(sequence)
    d = a_instance.method_IP(sequence, newx3, roll_width)
    e = a_instance.row_by_row(sequence)

    f = a_instance.offline(sequence)  # optimal result
    optimal = np.dot(multi, f)

    ratio1 += np.dot(multi, a) / optimal  # sto-planning
    ratio2 += np.dot(multi, b) / optimal  # bid-price
    ratio3 += np.dot(multi, c) / optimal  # DP1
    ratio4 += np.dot(multi, d) / optimal  # booking-limit
    ratio5 += np.dot(multi, e) / optimal  # FCFS
    accept_people += optimal
