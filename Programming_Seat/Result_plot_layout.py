import numpy as np
import matplotlib.pyplot as plt

# This function is used to load the data and plot the layout graphs.
# The specific parameters are as follows:

def plot_data(given_lines, roll_width, data):
    sd = 1
    t_value = data[0]
    people_value = data[1]
    occup_value = data[2]

    total_seat = np.sum(roll_width) - given_lines * sd

    for i in range(len(t_value)):
        if people_value[i]/100*total_seat - occup_value[i]/100*total_seat > 1:
            point = [t_value[0]+i-1, occup_value[i-1]]
            break

    plt.plot(t_value, people_value, 'b-', label = 'Without social distancing')
    plt.plot(t_value, occup_value, 'r--', label = 'With 1 social distancing')
    # plt.xlim((50, 250))
    # plt.ylim((0, 100))
    plt.xlabel('Periods')
    plt.ylabel('Percentage of total seats')
    
    point[1] = round(point[1], 2)
    plt.annotate(r'Gap $%s$' % str(point), xy = point, xytext = (point[0], point[1]), arrowprops = dict(facecolor = 'black', shrink = 0.1),)
    
    plt.legend()
    plt.savefig('Layout_large_not_rec.pdf')


# roll_width = np.ones(15) * 8
# roll_width = np.ones(10) * 21
# roll_width = np.ones(20) * 11
# roll_width = np.array([5, 7, 9, 11, 11])
# roll_width = np.ones(11) * 15
roll_width = np.array([15, 15, 15, 15, 15, 15, 17, 17, 17, 17, 17, 17, 17, 17, 17, 19])
# roll_width = np.array([17, 18, 19, 20, 21, 21, 22, 23, 24, 25])
given_lines = len(roll_width)

data = np.load('layout_large_not_rec.npy')
plot_data(given_lines, roll_width, data)
