import re
import numpy as np
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
    cnt = 0
    for i in range(len(data)):
        prob = re.findall(r"\d+\.?\d*", data[i][0])
        p = [float(i) for i in prob]
        # theta = p[-1]  # np.var  np.std
        gamma = np.dot(p, np.arange(1,5))
        gap_point = re.findall(r"\d+\.?\d*", data[i][1])
        period = int(gap_point[0])
        occu_rate = round(float(gap_point[1]), 2)
        result[cnt] = [gamma, period, occu_rate]
        cnt += 1
    return result

result = gamma(data)

#  根据已有数据拟合 + 画出拟合之后的图

data_x = result[:, 0]

# ####### Gap point estimation
# ####### c1 * L
# ########  x1  = 201.1413  c1 = 0.9578
# data_x1 = 1/(data_x+1)
# data_y1 = result[:, 1]

# ####### Occupancy estimations
# ####### c2 * L/(L-N) = c2 * 21/20
##########  x1 = 100.5431  c2 = 95.76

data_x2 = data_x/(data_x+1) 
data_y2 = result[:, 2]
mod = sm.OLS(data_y2, data_x2)  # 无截距项

# mod = sm.OLS(data_y1, data_x1)  # 无截距项

res = mod.fit()
print(res.summary())