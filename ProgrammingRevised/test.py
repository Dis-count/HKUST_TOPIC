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
    result = np.zeros((len(data),3))
    cnt = 0
    for i in range(len(data)):
        prob = re.findall(r"\d+\.?\d*", data[i][0])
        p = [float(i) for i in prob]
        p = np.dot(p, np.arange(1,5))
        gap_point = re.findall(r"\d+\.?\d*", data[i][1])
        period = int(gap_point[0])
        occu_rate = round(float(gap_point[1]), 2)
        result[cnt] = [p, period, occu_rate]
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

data_x1 = 1/(data_x+1)
data_y1 = result[:, 1]

data_x2 = data_x/(data_x+1)
data_y2 = result[:, 2]

# mod = sm.OLS(data_y, sm.add_constant(data_x2))  # 需要用sm.add_constant 手动添加截距项
mod = sm.OLS(data_y2, data_x2)  # 无截距项

res = mod.fit()
print(res.summary())

