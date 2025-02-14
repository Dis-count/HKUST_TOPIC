import numpy as np
import matplotlib.pyplot as plt

# This function is used to load the data and plot the layout graphs.
# The specific parameters are as follows:

def plot_data(given_lines, roll_width, data, file):
    sd = 1
    t_value = data[0]
    people_value = data[1]
    occup_value = data[2]

    total_seat = np.sum(roll_width) - given_lines * sd
    point = [0, 0]
    for i in range(len(t_value)):
        if people_value[i]/100 * total_seat - occup_value[i]/100 * total_seat > 1:
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
    plt.savefig(file + '.pdf')
    plt.close()
    # plt.savefig('layout_KTT_TS.pdf')


# file = 'layout_HKFAC', 'layout_KTT_TS'

# 'HK_FAC': np.array([17, 18, 18, 18, 18, 18, 18, 8]),
# 'KTT_TS': np.array([7, 9, 7, 10, 7, 11, 8, 11, 9, 7, 10, 7, 11, 7, 13, 8])
# 'SWHCC': np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
# 'SWCC': np.array([3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])
# 'NCWCC': np.array([13, 21, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 21, 13])


# roll_width = np.array([17, 18, 18, 18, 18, 18, 18, 8])
# np.array([7, 9, 7, 10, 7, 11, 8, 11, 9, 7, 10, 7, 11, 7, 13, 8])
# np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
# np.array([3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])
# np.array([13, 21, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 21, 13])

dic =   {'HKFAC': np.array([17, 18, 18, 18, 18, 18, 18, 8]),
         'KTT_TS': np.array([7, 9, 7, 10, 7, 11, 8, 11, 9, 7, 10, 7, 11, 7, 13, 8]),
         'SWHCC': np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
         'SWCC': np.array([3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]),
        'NCWCC': np.array([13, 21, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 21, 13])}

for shape, roll_width in dic.items():
    given_lines = len(roll_width)
    data = np.load('layout_' + shape + '.npy')
    plot_data(given_lines, roll_width, data, shape)
