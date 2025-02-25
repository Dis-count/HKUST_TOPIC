# Return the maximum achievable occupancy rate

import numpy as np

def largest(L, M, sd):
    q = int(L/(M+sd))
    r = L - (M+sd)*q

    return q*M + max(r-sd, 0)

def max_achieve(layout, M, sd):
    num = 0
    for i in layout:
        num += largest(i, M, sd)
    total_L = sum(layout)
    
    return num/(total_L - len(layout)*sd)


if __name__ == "__main__":
    # layout_dic = {'HK_FAC': np.array([17, 18, 18, 18, 18, 18, 18, 8]),
    #               'KTT_TS': np.array([7, 9, 7, 10, 7, 11, 8, 11, 9, 7, 10, 7, 11, 7, 13, 8]),
    #               'SWHCC': np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
    #               'SWCC': np.array([3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]),
    #               'NCWCC': np.array([13, 21, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 21, 13])}
    # for shape, roll_width in layout_dic.items():
        
    #     rate  = max_achieve(roll_width, 4, 1)
    #     print(shape + '\'s rate:' + str(rate))

    default_layout = np.ones(10)* 21

    rate = max_achieve(default_layout, 2, 1)

    print('default layout\'s layout is ' + str(rate))