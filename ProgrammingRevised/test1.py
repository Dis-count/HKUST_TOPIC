import re
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = [[0 for i in range(2)] for j in range(80)]

with open("Periods_layout.txt", 'r') as f:
    lines = f.readlines()[1:]
    cnt = 0
    num = 0
    for line in lines:
        if len(line) == 3:
            continue

        a = line.split('\t')
        if num % 2 == 0:
            data[cnt][0] = a[0]
        else:
            data[cnt][1] = a[0]
            cnt += 1
        num += 1


def gamma(data):
    result = np.zeros((len(data),3))
    cnt = 0
    for i in range(len(data)):
        prob = re.findall(r"\d+\.?\d*", data[i][0])
        p_gamma = round(float(prob[0]), 2)
        gap_point = re.findall(r"\d+\.?\d*", data[i][1])
        period = int(gap_point[0])
        occu_rate = round(float(gap_point[1]), 2)
        result[cnt] = [p_gamma, period, occu_rate]
        cnt += 1
    return result

result = gamma(data)

# plt.scatter(result[60:80, 0], result[60:80, 1], c="blue")
# plt.scatter(result[60:80, 0], result[60:80, 2], c="red")

# x = np.arange(1.5, 3.5, 0.01)
# y1 = 201.0395 /(x+1)
# y2 = 95.7992 * x/(x+1)


# plt.plot(x, y1, label="Blue_estimated")
# plt.plot(x, y2, label="Red_estimated")

# plt.xlabel('Gamma')
# plt.ylabel('Percentage of total seats/Periods')

# plt.legend()
# plt.show()


#  根据已有数据拟合 + 画出拟合之后的图
#  Result 6-3

# data_x = result[0:20, 0]

# data_x = result[20:40, 0]
# data_x = result[40:60, 0]
data_x = result[60:80, 0]

# data_x1 = 1/(data_x+1)
# data_y1 = result[0:20, 1]
# data_y1 = result[60:80, 1]

data_x2 = data_x/(data_x+1)
data_y2 = result[60:80, 2]

# mod = sm.OLS(data_y, sm.add_constant(data_x2))  # 需要用sm.add_constant 手动添加截距项


# mod = sm.OLS(data_y1, data_x1)  # 无截距项

mod = sm.OLS(data_y2, data_x2)

res = mod.fit()
print(res.summary())


# 151.6723/160   100.9988

# 200.6338   100.2097 

# 251.1247   100.0757

# 299.0610   99.8912
