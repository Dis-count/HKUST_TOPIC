import numpy as np
import matplotlib.pyplot as plt

# This function is used to load the data and plot the corrsponding graphs.
# The specific parameters are as follows:
# delta = 0, 1, 2

# 设置字体和 LaTeX 支持
# plt.rcParams['mathtext.fontset'] = 'cm'  # 使用 Computer Modern 字体渲染数学符号

def plot_data(given_lines, roll_width, data):
    sd = 1    
    t_value = data[0]
    people_value = data[1]
    occup_1 = data[2]
    occup_2 = data[3]

    total_seat = np.sum(roll_width) - given_lines * sd

    for i in range(len(t_value)):
        if people_value[i]/100*total_seat - occup_1[i]/100*total_seat > 1:
            point_1 = [t_value[0]+i-1, occup_1[i-1]]
            break
    for i in range(len(t_value)):
        if people_value[i]*total_seat - occup_2[i]*total_seat > 100:
            point_2 = [t_value[0]+i-1, occup_2[i-1]]
            break

    plt.plot(t_value, people_value, 'b-', label= 'Without social distancing')
    plt.plot(t_value, occup_1, 'r--', label=r'$\delta = 1$')
    plt.plot(t_value, occup_2, 'g-.', label=r'$\delta = 2$')

    plt.xlabel('Period')
    plt.ylabel('Occupancy rate (%)')

    point_1[1] = round(point_1[1], 1)
    plt.annotate(r'Threshold $%s$' % str(point_1), xy=point_1, xytext=(
        point_1[0]+10, point_1[1]), arrowprops = dict(facecolor='red', shrink=0.2),)

    point_2[1] = round(point_2[1], 1)
    plt.annotate(r'Threshold $%s$' % str(point_2), xy = point_2, xytext = (point_2[0]+10, point_2[1]), arrowprops = dict(facecolor = 'green', shrink = 0.2),)

    plt.axhline(y = 80, xmin = 0, xmax = 1, color = 'red', linestyle=':')
    plt.axhline(y = 70, xmin = 0, xmax = 1, color = 'green', linestyle=':')

    plt.legend()
    plt.savefig('distance_M2.pdf')

given_lines = 10
roll_width = np.ones(given_lines) * 21
data = np.load('data_distances_012_M2.npy')
plot_data(given_lines, roll_width, data)
