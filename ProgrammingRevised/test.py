import re
import numpy as np
import matplotlib.pyplot as plt

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
# print(result[:, 0])

plt.scatter(result[:, 0], result[:, 1], c="blue")
plt.scatter(result[:, 0], result[:, 2], c="red")

plt.xlabel('Gamma')
plt.ylabel('Percentage of total seats/Periods')

plt.legend()
plt.show()


