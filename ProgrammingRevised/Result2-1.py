import gurobipy as grb
from gurobipy import GRB
import numpy as np
import time
from SamplingMethodNew import samplingmethod1
from Mist import generate_sequence, decision1
from collections import Counter
from Method3 import deterministicModel
from Method1 import stochasticModel

# This function uses benders' decomposition to solve stochastic Model directly.
# And give the once decision.

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    number_period = 55
    given_lines = 6
    np.random.seed(0)

    probab = [0.4, 0.4, 0.1, 0.1]
    sam = samplingmethod1(I, num_sample, number_period, probab)

    dw, prop = sam.get_prob()
    W = len(dw)

    # roll_width = np.arange(21, 21 + given_lines)
    roll_width = np.ones(given_lines) * 21
    total_seat = np.sum(roll_width)

    demand_width_array = np.arange(2, 2+I)

    sequence = generate_sequence(number_period, probab)

    my = stochasticModel(roll_width, given_lines,
                         demand_width_array, W, I, prop, dw)

    ini_demand, upperbound = my.solveBenders(eps=1e-4, maxit=20)

    ini_demand = np.ceil(ini_demand)

    deter = deterministicModel(roll_width, given_lines, demand_width_array, I)

    ini_demand, _ = deter.IP_formulation(np.zeros(I), ini_demand)
    ini_demand, _ = deter.IP_formulation(ini_demand, np.zeros(I))

    decision_list = decision1(sequence, ini_demand, probab)

    sequence = [i-1 for i in sequence if i > 0]
    total_people = np.dot(sequence, decision_list)
    final_demand = np.array(sequence) * np.array(decision_list)
    print(f'The total seats: {total_seat}')
    print(f'The total people:{total_people}')
    print(Counter(final_demand))
