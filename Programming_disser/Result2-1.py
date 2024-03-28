import gurobipy as grb
from gurobipy import GRB
import numpy as np
from SamplingMethodSto import samplingmethod1
from Mist import generate_sequence, decision1
from Method2 import originalModel
from Method10 import deterministicModel
from Comparison import CompareMethods
import time
import copy

# This function uses benders' decomposition to solve stochastic Model directly.
# And give the decision under the fixed seat planning.

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    period_range = range(140, 201, 20)
    given_lines = 20
    sd = 1
    probab = [0.25, 0.25, 0.25, 0.25]
    roll_width = np.ones(given_lines) * 21
    demand_width_array = np.arange(1+sd, 1+sd+I)

    filename = 'fixed_seat' + str(time.time()) + '.txt'
    my_file = open(filename, 'w')
    my_file.write('Run Start Time: ' + str(time.ctime()) + '\n')

    for number_period in period_range:
        my_file.write('The number of periods: \t' + str(number_period) + '\n')

        ex_demand = [i * number_period for i in probab]
        der = deterministicModel(roll_width, given_lines, demand_width_array, I, sd)
        ex_demand, _ = der.IP_formulation(np.zeros(I), ex_demand)

        sam = samplingmethod1(I, num_sample, number_period, probab, sd)

        dw, prop = sam.get_prob()
        W = len(dw)

        count = 50
        ratio = 0
        ratio1 = 0
        optimal = 0
        multi = np.arange(1, I+1)
        my = originalModel(roll_width, given_lines,
                                demand_width_array, W, I, prop, dw, sd)

        ini_demand, _ = my.solveModelGurobi()
        
        for i in range(count):
            ini_demand1 = copy.deepcopy(ini_demand)
            ex_demand1 = copy.deepcopy(ex_demand)
            sequence = generate_sequence(number_period, probab, sd)
            # sequence1 = copy.deepcopy(sequence)
            a = CompareMethods(roll_width, given_lines, I,
                            probab, number_period, num_sample, sd)
            f = a.offline(sequence)
            optimal = np.dot(multi, f)

            decision_list = decision1(sequence, ini_demand1, probab, sd)
            decision_list1 = decision1(sequence, ex_demand1, probab, sd)
            sequence = [j-sd for j in sequence]

            avg  = np.dot(sequence, decision_list)
            avg1 = np.dot(sequence, decision_list1)

            ratio += avg/optimal
            ratio1 += avg1/optimal

        my_file.write('SSP: %.2f ;' % (ratio/count*100))
        my_file.write('Expected: %.2f ;\n' % (ratio1/count*100))

    my_file.close()
