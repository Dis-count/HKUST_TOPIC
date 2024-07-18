# import gurobipy as grb
# from gurobipy import GRB
import statsmodels.api as sm
import numpy as np
from Comparison import CompareMethods
import time
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    given_lines = 10
    # np.random.seed(i)
    p = prop_list()
    p_len = len(p)

    c_value = np.zeros(p_len)
    people_value = np.zeros(p_len)
    optimal_value = np.zeros(p_len)
    
    begin_time = time.time()
    filename = 'different_c' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    count = 50
    for ind_probab, probab in enumerate(p):

        my_file.write('probabilities: \t' + str(probab) + '\n')
        roll_width = np.ones(given_lines) * 21
        total_seat = np.sum(roll_width)
        
        multi = np.arange(1, I+1)
        c_p = multi @ probab
        c_value[ind_probab] = c_p
        # num_period = round(total_seat/(c_p + 1))
        num_period = 90
        a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)

        ratio_sto = 0
        accept_people = 0
        # num_people = 0

        for j in range(count):
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
            # total_people = sum(sequence) - num_period

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            g = a_instance.method_new(sequence, newx4, roll_width)
            # num_people += total_people

            ratio_sto += np.dot(multi, g)
            accept_people += optimal

        optimal_value[ind_probab] = accept_people/count
        people_value[ind_probab] = ratio_sto/count
        my_file.write('Result: %.2f \n' % (ratio_sto/count))


    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t %f \n' % run_time)

    max_acc = 0
    max_ind = 0
    min_acc = 200
    min_ind = 0
    for i in range(p_len):
        if people_value[i] > max_acc:
            max_acc = people_value[i]
            max_ind = i
        if people_value[i] < min_acc:
            min_acc = people_value[i]
            min_ind = i
    my_file.write('Minimum Prob \t' + str(p[min_ind]) + '\n')
    my_file.write('Maximum Prob \t' + str(p[max_ind]))
    
    my_file.close()
