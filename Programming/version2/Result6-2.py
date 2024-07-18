# import gurobipy as grb
# from gurobipy import GRB
import statsmodels.api as sm
import numpy as np
from Comparison import CompareMethods
import time
import matplotlib.pyplot as plt

def prop_list():
    x1 = np.arange(0.05, 1, 0.05)
    x2 = np.arange(0.05, 1, 0.05)
    x3 = np.arange(0.05, 1, 0.05)

    p = np.zeros((len(x1)*len(x2)*len(x3), 4))
    t = 0

    for i in x1:
        for j in x2:
            for k in x3:
                if 1 - i - j - k >0:
                    p[t] = [i, j, k, 1 - i - j - k]
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
        num_period = round(total_seat/(c_p + 1))

        a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample)

        ratio_sto = 0
        accept_people = 0
        # num_people = 0

        for j in range(count):
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
            # total_people = sum(sequence) - num_period

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            g = a_instance.method_new(sequence, ini_demand, newx4, roll_width)
            # num_people += total_people

            ratio_sto += np.dot(multi, g)
            accept_people += optimal

        optimal_value[ind_probab] = accept_people/count
        people_value[ind_probab] = ratio_sto/count
        my_file.write('Result: %.2f \n' % (ratio_sto/count))

        # my_file.write('Number of accepted people: %.2f \t' %(accept_people/count))
        # my_file.write('Number of people: %.2f \n' % (num_people/count))
        
        # if c_p < 3.2:
        #     occup_value = sum(roll_width) * (c_p/(c_p+1))
        # else:
        #     occup_value = 160
        # my_file.write('Mean estimation: %.2f \n' % occup_value)

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t %f \n' % run_time)

    occup_value = np.zeros(p_len)
    diff = np.zeros(p_len)
    ratio = 0
    for i in range(p_len):
        if c_value[i] <= 3.2:
            occup_value[i] = sum(roll_width) * (c_value[i]/(c_value[i]+1))
        else:
            occup_value[i] = 160

        diff[i] = occup_value[i]- people_value[i]

    # for i in range(p_len):
    #     if c_value[i] ==2:
            
        # if abs(diff[i]) >= 3:
        #     ratio += 1
        #     my_file.write('Deviated probability: ' + str(p[i]) + '\t')
        #     my_file.write('Deviated value: %.2f \n' % diff[i])
    
    # print(sum(abs(diff) >= 4)/p_len)
    # print(sum(abs(diff) >= 3)/p_len)
    # print(sum(abs(diff) >= 2)/p_len)
    # print(sum(abs(diff) >= 1)/p_len)
    # plt.hist(diff, bins=20, color='red', alpha=0.75)
    # plt.title('Difference Distribution')
    # plt.show()

    gamma = c_value/(c_value+1)
    plt.scatter(gamma, people_value, c = "blue")
    plt.scatter(gamma, occup_value, c = "red")
    plt.show()

my_file.close()

    # plt.scatter(c_value, people_value, c = "blue")
    # plt.scatter(c_value, occup_value, c = "red")