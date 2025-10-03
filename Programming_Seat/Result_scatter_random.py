import re
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = [[0 for _ in range(2)] for _ in range(200)]

with open("Random_200.txt", 'r') as f:
    lines = f.readlines()[1:]
    cnt = 0
    for line in lines:
        a = line.split('\t')
        data[cnt] = a
        cnt += 1

def gamma(data):
    result = np.zeros((len(data), 3))
    prop_set = np.zeros((len(data), 4))
    cnt = 0
    for i in range(len(data)):
        prob = re.findall(r"\d+\.?\d*", data[i][0])
        p = [float(i) for i in prob]
        prop_set[cnt] = p
        # theta = p[-1]  # np.var  np.std
        gamma = np.dot(p, np.arange(1,5))
        gap_point = re.findall(r"\d+\.?\d*", data[i][1])
        period = int(gap_point[0])
        occu_rate = round(float(gap_point[1]), 2)
        result[cnt] = [gamma, period, occu_rate]
        cnt += 1
    return result, prop_set

result, prop_set = gamma(data)


x = np.arange(1.4, 3.6, 0.01)

y1 = 201.14 /(x+1)
y2 = 100.54 * x/(x+1)

# for i in range(200):
#     x_2 = result[i][0]
#     if abs(100.54 * x_2/(x_2+1) - result[i][2]) > 3:
#         print(prop_set[i])
#         print(result[i][2])
#         print(result[i][1])
#         print(result[i][0])

# [0.2  0.05 0.1  0.65]
# 73.27
# 46.0
# 3.2

plt.xlabel('Gamma ($\gamma$)', fontsize = 24)

plt.scatter(result[:, 0], result[:, 1], c = "blue")
plt.plot(x, y1, label= "Blue_estimated")
plt.ylabel('Period ($T$)', fontsize= 24)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.twinx()

plt.scatter(result[:, 0], result[:, 2], c="red")
plt.plot(x, y2, 'r--', label= "Red_estimated")
plt.ylabel('Occupancy rate (%)', fontsize = 24)


plt.text(1.8, 76, "Estimate of $q^{th}$", size = 24, alpha = 1)

plt.text(3.2, 71, r'Estimate of $\rho^{th}$', size = 24, alpha = 1)

plt.yticks(np.arange(60, 80, 5), fontsize=24)
plt.show()


#  根据已有数据拟合 + 画出拟合之后的图
#  Result 6-3

# data_x = result[:, 0]

# ####### Gap point estimation
#  c1 * L
# data_x1 = 1/(data_x+1)
# data_y1 = result[:, 1]

# ####### Occupancy estimations
# c2 * L/(L-N) = c2* 21/20
# data_x2 = data_x/(data_x+1) 
# data_y2 = result[:, 2]

# mod = sm.OLS(data_y2, sm.add_constant(data_x2))  # 需要用sm.add_constant 手动添加截距项

# ############ When considering multiple independent variables 
# ############ Not useful fitting with variance and prob_1
# theta = result[:, -1]
# datax1 = np.array([data_x2]).T
# theta = np.array([theta]).T
# data_x1 = np.concatenate((datax1, theta), axis=1)


# mod = sm.OLS(data_y1, data_x1)  # 无截距项

# res = mod.fit()
# print(res.summary())

# print(a.reshape(-1, 1))
# print(np.concatenate((a.T, b.T), axis=1))

# const         -1.2799      0.446     -2.867      0.005      -2.160      -0.399
# x1           203.6197      1.517    134.253      0.000     200.629     206.611
# x2             6.7697      2.771      2.443      0.015       1.305      12.234

# const         -4.6073      0.456    -10.107      0.000      -5.506      -3.708
# x1           219.7493      1.800    122.068      0.000     216.199     223.300
# x2            -4.7545      0.408    -11.657      0.000      -5.559      -3.950