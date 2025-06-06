import gurobipy as grb
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from obj_three import originalModel
import uuid

# This function is used to plot the graph of the optimal scheduling interval with overlapping time constraints.

# Use SSA to simulate the problem

def interval_graph(delta):
    # Each row of the delta is the optimal schedule interval.

    indices = range(1, len(delta[0]) + 1)

    for i, y in enumerate(delta):
        plt.plot(indices, y, label=f'y{i+1}')

    # plt.plot(indices, delta, 'bo-')

    plt.xlabel('Customers')
    plt.ylabel('Schedule Interval')
    # plt.legend()
    plt.xticks(indices)
    plt.grid(axis='x')
    unique_id = uuid.uuid4().hex[:8]  # 取 UUID 的前 8 位
    filename = f"plot_{unique_id}.pdf"  # 例如：plot_3a4b5c6d.pdf
    plt.savefig(filename)
    plt.close()
    # plt.show()

if __name__ == "__main__":
    num_sample = 5000
    people_num = 6
    overlapping_thres = np.zeros(people_num)
    overlapping_thres[0] = 50  # one person waiting
    overlapping_thres[1] = 20  # two person waiting
    overlapping_thres[2] = 5
    T = 125
    c_i = 5
    c_w = 1
    c_o = 0

    # np.random.seed(0)

    # test normal distribution

    for i in range(30):
        # zeta = np.random.exponential(20, [people_num, num_sample])
        # zeta = np.floor(zeta).astype(int)
        # zeta = np.clip(zeta, None, 60)

        mu, sigma = 20, 20
        a, b = (0 - mu) / sigma, np.inf
        zeta = truncnorm.rvs(a, b, loc = mu, scale = sigma, size = (people_num, num_sample))
        zeta = np.floor(zeta).astype(int)

        appointment = originalModel(people_num, num_sample, zeta, overlapping_thres, T, c_i, c_w, c_o)
        delta = np.zeros((6, people_num-1))

        ind = 0 
        for thres in [12, 10, 8, 6]:
            appointment.wij[1] = thres  # 直接修改 T
            delta[ind,:] = appointment.overlapping()
            ind += 1

        interval_graph(delta)
