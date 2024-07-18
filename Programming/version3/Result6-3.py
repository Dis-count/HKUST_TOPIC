# import gurobipy as grb
# from gurobipy import GRB
import random
import statsmodels.api as sm
import numpy as np
from Comparison import CompareMethods
import time
import matplotlib.pyplot as plt
import pandas as pd

# Do the linear regression for gamma

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
    sd = 1
    p = prop_list()
    p_len = len(p)
    cp_len = 0
    for probab in p:
        c_p = np.arange(1, I+1) @ probab
        if c_p <= 3.2:
            cp_len += 1

    c_value = np.zeros(p_len)
    people_value = np.zeros(p_len)

    begin_time = time.time()
    filename = 'different_c' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    my_cnt = 0
    cnt = 0
    count = 20
    dataset = np.zeros((cp_len * count, 6))

    for ind_probab, probab in enumerate(p):

        my_file.write('probabilities: \t' + str(probab) + '\n')

        roll_width = np.ones(given_lines) * 21
        total_seat = np.sum(roll_width)
        num_period = round(total_seat/(c_p + sd))
        a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)

        ratio_sto = 0
        accept_people = 0
        # num_people = 0

        multi = np.arange(1, I+1)
        c_p = multi @ probab
        # c_value[cnt] = c_p
        gamma = c_p/(c_p + sd) * total_seat

        for j in range(count):
            sequence, ini_demand, ini_demand3, newx3, newx4 = a_instance.random_generate()
            # total_people = sum(sequence) - num_period

            f = a_instance.offline(sequence)  # optimal result
            optimal = np.dot(multi, f)

            g = a_instance.method_new(sequence, ini_demand, newx4, roll_width)
            sto = np.dot(multi, g)
            # num_people += total_people
            ratio_sto += sto
            accept_people += optimal

        if c_p <= 3.2:
            data = np.append(probab, gamma)
            data = np.append(data, ratio_sto/count)
            dataset[my_cnt, :] = data
            my_cnt += 1

        my_file.write('Result: %.2f \n' % (ratio_sto/count))
        # my_file.write('Number of accepted people: %.2f \t' %(accept_people/count))
        # my_file.write('Number of people: %.2f \n' % (num_people/count))
        
        people_value[cnt] = ratio_sto/count
        cnt += 1

        # if c_p < 3.2:
        #     occup_value = sum(roll_width) * (c_p/(c_p+1))
        # else:
        #     occup_value = 160
        # my_file.write('Mean estimation: %.2f \n' % occup_value)

    run_time = time.time() - begin_time
    my_file.write('Total Runtime\t %f \n' % run_time)

    # occup_value = np.zeros(p_len)
    # for i in range(p_len):
    #     if c_value[i] < 3.2:
    #         occup_value[i] = sum(roll_width) * (c_value[i]/(c_value[i]+1))
    #     else:
    #         occup_value[i] = 160

        # diff = occup_value[i]- people_value[i]
        # if abs(diff) >= 3:
        #     my_file.write('Deviated probability: ' + str(p[i]) + '\t')
        #     my_file.write('Deviated value: %.2f \n' % diff)

    # plt.scatter(c_value, people_value, c = "blue")
    # plt.scatter(c_value, occup_value, c = "red")
    # plt.show()

dataA = pd.DataFrame(dataset, columns=["p1", "p2", "p3", "p4", 'gamma', 'y'])

# data_x = dataA[['p1','p2','p3']]
data_x = dataA['gamma']

data_y = dataA['y']

mod = sm.OLS(data_y, sm.add_constant(data_x))  # 需要用sm.add_constant 手动添加截距项
res = mod.fit()

my_file.write(str(res.summary()))
my_file.close()

# writer = pd.ExcelWriter('A50.xlsx')		# 写入Excel文件
# dataA.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer.save()
# writer.close()
