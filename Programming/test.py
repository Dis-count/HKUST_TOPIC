from Method1 import stochasticModel
from SamplingMethod import samplingmethod
import numpy as np

num_sample = 30  # the number of scenarios
I = 4  # the number of group types
number_period = 10
given_lines = 8
np.random.seed(0)

probab = [0.4, 0.4, 0.1, 0.1]
sam = samplingmethod(I, num_sample, number_period, probab)

dw, prop = sam.get_prob()
W = len(dw)

roll_width = np.ones(given_lines)* 20

total_seat = np.sum(roll_width)

demand_width_array = np.arange(2, 2+I)


my = stochasticModel(roll_width, given_lines,
                     demand_width_array, W, I, prop, dw)

ini_demand, upperbound = my.solveBenders(eps=1e-4, maxit=20)

print(ini_demand)