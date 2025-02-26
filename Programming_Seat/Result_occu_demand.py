import numpy as np
import matplotlib.pyplot as plt

# This function is used to give the occupancy over demand

# [0.25, 0.3, 0.25, 0.2]

def plot_data(data, option):
    probab = [0.12, 0.5, 0.13, 0.25]
    gamma = np.dot(probab, np.arange(1, 5))
    given_lines = 10
    roll_width = np.ones(given_lines) * 21
    sd = 1
    total_seat = np.sum(roll_width) - given_lines * sd

    t_value = data[0]
    people_value = data[1]
    occup_value = data[2]

    for i in range(len(t_value)):
        if people_value[i]/100*total_seat - occup_value[i]/100*total_seat > 1:
            point = [t_value[0]+i-1, occup_value[i-1]]
            break

    if option:
        plt.plot(t_value, people_value, 'b-', label='Without social distancing')
        plt.plot(t_value, occup_value, 'r--', label='With 1 social distancing')
        plt.xlim((40, 100))
        plt.ylim((40, 100))
        plt.xlabel('Period')
        plt.ylabel('Percentage of accepted individuals relative to total seats')
        point[1] = round(point[1], 1)

        plt.annotate(r'Gap $%s$' % str(point), xy=(point[0], point[1]), xytext=(
            point[0] + 5, point[1]-10), arrowprops=dict(facecolor='black', shrink=0.1),)

        plt.axhline(y = 80, xmin = 0, xmax = 1, color = 'green', linestyle = '--')
        # plt.axhline(y = 65, xmin = 0, xmax = 1, color = 'purple', linestyle='--')
        plt.axhline(y = 71.8, xmin = 0, xmax = 1, color = 'purple', linestyle='--')

        # plt.annotate(r'80%' , xy=(90, 80), xytext=(80, 90), color='red', arrowprops=dict(facecolor='black', shrink=0.1),)

        my_x_ticks = np.arange(40, 100, 10)
        plt.xticks(my_x_ticks)
        plt.legend()
        graphname = 'occu_demand_group4.pdf'
        plt.savefig(graphname)

    else:
        plt.plot(t_value* gamma/total_seat*100, people_value, 'b-', label='Without social distancing')
        plt.plot(t_value* gamma/total_seat*100, occup_value, 'r--', label='With 1 social distancing')
        plt.xlim((50, 110))
        plt.ylim((50, 110))
        plt.xlabel('Percentage of expected demand relative to total seats')
        plt.ylabel('Percentage of accepted individuals relative to total seats')
        point[1] = round(point[1], 1)
        # plt.axvline(x = 60, ymin = 0, ymax = 1/6, color = 'green', linestyle='--')

        plt.axvline(x = 71.8, ymin = 0, ymax = 0.36, color = 'purple', linestyle='--')
        plt.axvline(x = 80,  ymin = 0, ymax = 1/2, color = 'green', linestyle='--')
        # plt.axvline(90, ymin = 0, ymax = 2/3, color = 'purple', linestyle='--')
        # plt.axvline(100, ymin = 0, ymax = 5/6-0.02, color = 'purple', linestyle='--')

        my_x_ticks = np.arange(50, 110, 10)
        plt.xticks(my_x_ticks)
        plt.legend()
        graphname = 'occu_gamma_group4.pdf'
        plt.savefig(graphname)


data = np.load('data_group_4.npy')
# period
option = 1

# gamma
option = 0

plot_data(data, 0)
