import re
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = [[0 for i in range(2)] for j in range(200)]

with open("Periods_202.txt", 'r') as f:
    lines = f.readlines()[1:]
    cnt = 0
    for line in lines:
        a = line.split('\t')
        data[cnt] = a
        cnt += 1


def gamma(data):
    result = np.zeros((len(data),4))
    cnt = 0
    for i in range(len(data)):
        prob = re.findall(r"\d+\.?\d*", data[i][0])
        p = [float(i) for i in prob]
        theta = p[0]  # np.var  np.std
        p = np.dot(p, np.arange(1,5))
        gap_point = re.findall(r"\d+\.?\d*", data[i][1])
        period = int(gap_point[0])
        occu_rate = round(float(gap_point[1]), 2)
        result[cnt] = [p, period, occu_rate, theta]
        cnt += 1
    return result

result = gamma(data)

# plt.scatter(result[:, 0], result[:, 1], c="blue")
# plt.scatter(result[:, 0], result[:, 2], c="red")

# x = np.arange(1.5, 3.5, 0.01)
# y1 = 200.02 /(x+1)
# y2 = 95.475 * x/(x+1)

# plt.plot(x, y1, label="Blue_estimated")
# plt.plot(x, y2, label="Red_estimated")

# plt.xlabel('Gamma')
# plt.ylabel('Percentage of total seats/Periods')

# plt.legend()
# plt.show()


#  根据已有数据拟合 + 画出拟合之后的图
#  Result 6-3

data_x = result[:, 0]

theta = result[:, -1]

data_x1 = 1/(data_x+1)
data_y1 = result[:, 1]

data_x2 = data_x/(data_x+1)
data_y2 = result[:, 2]

# mod = sm.OLS(data_y, sm.add_constant(data_x2))  # 需要用sm.add_constant 手动添加截距项

datax1 = np.array([data_x1]).T
theta = np.array([theta]).T

data_x1 = np.concatenate((datax1, theta), axis=1)

mod = sm.OLS(data_y1, data_x1)  # 无截距项

res = mod.fit()
print(res.summary())

# print(a.reshape(-1, 1))

# print(np.concatenate((a.T, b.T), axis=1))


# const         -1.2799      0.446     -2.867      0.005      -2.160      -0.399
# x1           203.6197      1.517    134.253      0.000     200.629     206.611
# x2             6.7697      2.771      2.443      0.015       1.305      12.234

# const         -4.6073      0.456    -10.107      0.000      -5.506      -3.708
# x1           219.7493      1.800    122.068      0.000     216.199     223.300
# x2            -4.7545      0.408    -11.657      0.000      -5.559      -3.950