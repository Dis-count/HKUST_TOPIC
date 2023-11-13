import re
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = [[0 for i in range(2)] for j in range(200)]

with open("Periods_1row.txt", 'r') as f:
    lines = f.readlines()[1:]
    cnt = 0
    for line in lines:
        a = line.split('\t')
        data[cnt] = a
        cnt += 1


def gamma(data):
    result = np.zeros((len(data),3))
    cnt = 0
    for i in range(len(data)):
        prob = re.findall(r"\d+\.?\d*", data[i][0])
        p = [float(i) for i in prob]
        p_gamma = np.dot(p, np.arange(1,5))
        if p_gamma == 3.05:
            print(p)
        gap_point = re.findall(r"\d+\.?\d*", data[i][1])
        period = int(gap_point[0])
        occu_rate = round(float(gap_point[1]), 2)
        result[cnt] = [p_gamma, period, occu_rate]
        cnt += 1
    return result

result = gamma(data)

plt.scatter(result[:, 0], result[:, 1], c="blue")
plt.scatter(result[:, 0], result[:, 2], c="red")

x = np.arange(1.2, 3.5, 0.01)
y1 = 200.9392 /(x+1)
y2 = 95.9972 * x/(x+1)

# 200.02
# 95.475

# 200.4914
# 95.7091

# 200.9392
# 95.9972

plt.plot(x, y1, label="Blue_estimated")
plt.plot(x, y2, label="Red_estimated")

plt.xlabel('Gamma')
plt.ylabel('Percentage of total seats/Periods')

plt.legend()
plt.show()


#  根据已有数据拟合 + 画出拟合之后的图
#  Result 6-3

# data_x = result[:, 0]


# data_x1 = 1/(data_x+1)
# data_y1 = result[:, 1]

# data_x2 = data_x/(data_x+1)
# data_y2 = result[:, 2]

# mod = sm.OLS(data_y, sm.add_constant(data_x2))  # 需要用sm.add_constant 手动添加截距项


# mod = sm.OLS(data_y1, data_x1)  # 无截距项

# mod = sm.OLS(data_y2, data_x2)

# res = mod.fit()
# print(res.summary())




# const         -1.2799      0.446     -2.867      0.005      -2.160      -0.399
# x1           203.6197      1.517    134.253      0.000     200.629     206.611
# x2             6.7697      2.771      2.443      0.015       1.305      12.234

# const         -4.6073      0.456    -10.107      0.000      -5.506      -3.708
# x1           219.7493      1.800    122.068      0.000     216.199     223.300
# x2            -4.7545      0.408    -11.657      0.000      -5.559      -3.950