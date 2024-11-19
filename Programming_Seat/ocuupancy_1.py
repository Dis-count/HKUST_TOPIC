import numpy as np
import matplotlib.pyplot as plt

probab = [0.25, 0.25, 0.25, 0.25]
gamma = np.dot(probab, np.arange(1, 5))

given_lines = 10
roll_width = np.ones(given_lines) * 21
sd = 1

total_seat = np.sum(roll_width) - given_lines * sd

t_value = np.load("t_value.npy")
people_value = np.load("peo_value.npy")
occup_value = np.load("occup_value.npy")
point=[58, 71.3]

plt.plot(t_value, people_value,
         'b-', label='Without social distancing')
plt.plot(t_value, occup_value,
         'r--', label='With social distancing')
plt.xlim((30, 90))
plt.ylim((40, 100))
plt.xlabel('Period')
plt.ylabel('Percentage of accepted individuals relative to total seats')
point[1] = round(point[1], 2)
# plt.annotate(r'Gap $%s$' % str(point), xy=point, xytext=(
# point[0] + 10, point[1]-20), arrowprops=dict(facecolor='black', shrink=0.1),)

plt.annotate(r'Gap $%s$' % str(point), xy=(point[0], point[1]), xytext=(
    point[0] + 5, point[1]-10), arrowprops=dict(facecolor='black', shrink=0.1),)

plt.axhline(y=80, xmin=0, xmax=1, color='green', linestyle='--')

# plt.annotate(r'80%' , xy=(90, 80), xytext=(80, 90), color='red', arrowprops=dict(facecolor='black', shrink=0.1),)

my_x_ticks = np.arange(30, 90, 10)
plt.xticks(my_x_ticks)
plt.legend()
graphname = './test' + str(probab[0]) + '.pdf'
plt.savefig(graphname)
plt.cla()
