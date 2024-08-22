import numpy as np
from SamplingMethodSto import samplingmethod1
from Method1 import stochasticModel
from Method10 import deterministicModel
from Mist import generate_sequence, decision1
import copy
from Mist import decisionSeveral, decisionOnce
import time
import matplotlib.pyplot as plt

# This function call different methods

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, num_sample, s):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.s = s
        self.demand_width_array = np.arange(1+s, 1+I+s)
        self.value_array = self.demand_width_array - self.s
        self.I = I   # number, Immutable object
        self.probab = probab
        self.num_period = num_period   # number, Immutable object
        self.num_sample = num_sample   # number, Immutable object

    def random_generate(self):
        sam = samplingmethod1(self.I, self.num_sample,
                             self.num_period, self.probab, self.s)

        dw, prop = sam.get_prob()
        W = len(dw)

        sequence = generate_sequence(self.num_period, self.probab, self.s)

        m1 = stochasticModel(self.roll_width, self.given_lines,
                             self.demand_width_array, W, self.I, prop, dw, self.s)

        ini_demand, _ = m1.solveBenders(eps=1e-4, maxit=20)

        deter = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)

        ini_demand, _ = deter.IP_formulation(np.zeros(self.I), ini_demand)
        ini_demand, _ = deter.IP_formulation(ini_demand, np.zeros(self.I))

        ini_demand1 = np.array(self.probab) * self.num_period

        ini_demand3, _ = deter.IP_formulation(np.zeros(self.I), ini_demand1)
        ini_demand3, _ = deter.IP_formulation(ini_demand3, np.zeros(self.I))

        return sequence, ini_demand, ini_demand3

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        sequence = [i-self.s for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I, self.s)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)

        return newd

    def offline1(self, sequence):
        # This function is to obtain the optimal decision without social distance.
        demand = np.zeros(self.I)
        sequence = [i-self.s for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel(self.roll_width, self.given_lines, self.demand_width_array-self.s, self.I, 0)
        newd, _ = test.IP_formulation(np.zeros(self.I), demand)
        
        return newd

    def method1(self, sequence, ini_demand):
        decision_list = decision1(sequence, ini_demand, self.probab)
        sequence = [i-self.s for i in sequence if i > 0]

        final_demand = np.array(sequence) * np.array(decision_list)
        # print('The result of Method 1--------------')
        final_demand = final_demand[final_demand!=0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand

    def result(self, sequence, ini_demand, ini_demand3):
        ini_demand4 = copy.deepcopy(ini_demand)

        final_demand1 = self.method1(sequence, ini_demand)

        final_demand3 = self.method4(sequence, ini_demand3)

        final_demand4 = self.method4(sequence, ini_demand4)

        return final_demand1, final_demand3, final_demand4

if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4   # the number of group types
    period_range = range(10,100,1)
    given_lines = 10
    sd = 1
    # np.random.seed(i)
    # probab = [0.3, 0.2, 0.2, 0.3]
    probab1 = [[0.2, 0.3, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25], [0.3, 0.2, 0.2, 0.3], [0.1, 0.4, 0.4, 0.1], [0.1, 0.5, 0.2, 0.2]]

    test = 0
    for probab in probab1:
        t_value = np.arange(10, 100, 1)
        people_value = np.zeros(len(period_range))
        occup_value = np.zeros(len(period_range))

        cnt = 0
        gap_if = True
        for num_period in period_range:
            roll_width = np.ones(given_lines) * 21
            total_seat = np.sum(roll_width)

            a_instance = CompareMethods(roll_width, given_lines, I, probab, num_period, num_sample, sd)

            M4 = 0
            accept_people = 0

            multi = np.arange(1, I+1)
            print(num_period)
            count = 50
            for j in range(count):
                sequence, ini_demand, ini_demand3 = a_instance.random_generate()

                # a, c, d = a_instance.result(sequence, ini_demand, ini_demand3)

                f = a_instance.offline1(sequence)  # optimal result
                optimal = np.dot(multi, f)

                d = a_instance.offline(sequence)

                # sequence, ini_demand = a_instance.random_generate()
                # a = a_instance.result(sequence, ini_demand)

                M4 += np.dot(multi, d)
                accept_people += optimal

            occup_value[cnt] = M4/count/total_seat * 100
            people_value[cnt] = accept_people/count/total_seat * 100
            if gap_if:
                if accept_people/count - M4/count > 1:
                    point = [num_period-1, occup_value[cnt-1]]
                    gap_if = False
            cnt += 1
        if test <= 0:
            plt.plot(t_value, people_value, 'b-', label='Without social distancing')
            test += 1
        plt.plot(t_value, occup_value, label = str(probab))
        point[1] = round(point[1], 2)
        # plt.annotate(r'Gap $%s$' % str(point), xy=point, xytext=(point[0]+10, point[1]-20), arrowprops=dict(facecolor='black', shrink=0.1),)
        print(point)
    plt.xlabel('Periods')
    plt.ylabel('Percentage of total seats')
    plt.legend()
    plt.show()
        

