import numpy as np
from Comparison1 import CompareMethods
import time
import matplotlib.pyplot as plt

# Plot arrival path of DSA
# Parameters

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = 70
    given_lines = 10
    probab = [0.25, 0.25, 0.25, 0.25]
    multi = np.arange(1, I+1)
    gamma = np.dot(probab, multi)
    np.random.seed(15)
    s = 1
    begin_time = time.time()
    t_value = np.arange(0, period_range+1, 1)

    roll_width = np.ones(given_lines) * 21
    total_seats = np.sum(roll_width)
    a_instance = CompareMethods(roll_width, given_lines, I, probab, period_range, num_sample, s)

    multi = np.arange(1, I+1)

    sequence, newx4 = a_instance.random_generate()

    accept_list = a_instance.method_new(sequence, newx4, roll_width)
    opt_demand = a_instance.offline(sequence)
    opt_list = []
    for i in sequence:
        if opt_demand[i-1-s] > 0:
            opt_list.append(1)
            opt_demand[i-1-s] -= 1 
        else:
            opt_list.append(0)
    accept_opt_seq = np.array(sequence) * np.array(opt_list)
    opt_seats = np.cumsum(accept_opt_seq)
    opt_remain_seats = total_seats - opt_seats
    opt_remain_seats = np.insert(opt_remain_seats, 0, total_seats)

    num_reject = sum(np.array(accept_list) == 0)
    print(num_reject)

    accept_seq = np.array(sequence) * np.array(accept_list)
    
    actual_seq = np.cumsum(sequence)
    act = np.insert(actual_seq, 0, 0)
    actual_seq = sum(sequence) - act

    accept_seats = np.cumsum(accept_seq)

    remain_seats = total_seats - accept_seats
    remain_seats = np.insert(remain_seats, 0, total_seats)
    # print(remain_seats)
    remain_demand = (period_range - t_value) * (gamma+s)

    plt.plot(t_value, remain_seats, 'b-', label='Number of remaining seats')
    plt.plot(t_value, remain_demand, 'r--', label='Expected future demand')
    plt.plot(t_value, actual_seq, '*', label='Actual remaining demand')
    plt.plot(t_value, opt_remain_seats, '*', label='Optimal remaining seats')
    plt.xlabel('Periods')
    plt.ylabel('Number of remaining seats/demand')
    plt.legend()
    plt.show()
