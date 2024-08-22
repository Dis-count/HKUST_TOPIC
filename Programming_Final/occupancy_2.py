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

plt.plot(t_value*gamma/total_seat*100, people_value,
         'b-', label='Without social distancing')
plt.plot(t_value*gamma/total_seat*100, occup_value,
         'r--', label='With social distancing')
plt.xlim((40, 100))
plt.ylim((40, 100))
plt.xlabel('Percentage of expected demand relative to total seats')
plt.ylabel('Percentage of accepted individuals relative to total seats')
point[1] = round(point[1], 2)
# plt.annotate(r'Gap $%s$' % str(point), xy=point, xytext=(
# point[0] + 10, point[1]-20), arrowprops=dict(facecolor='black', shrink=0.1),)

plt.axvline(x=60, ymin=0, ymax=1/3, color='green', linestyle='--')
plt.axvline(x = 70, ymin=0, ymax=0.50, color='green', linestyle='--')
plt.axvline(x=80,  ymin=0, ymax=2/3, color='purple', linestyle='--')
plt.axvline(90, ymin=0, ymax=5/6, color='purple', linestyle='--')

my_x_ticks = np.arange(40, 100, 10)
plt.xticks(my_x_ticks)
plt.legend()
graphname = './test' + str(probab[0]) + '.pdf'
plt.savefig(graphname)
plt.cla()
