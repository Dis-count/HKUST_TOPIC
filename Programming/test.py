import numpy as np
from SamplingMethod import samplingmethod
from Method1 import stochasticModel
from Method4 import deterministicModel
from Mist import generate_sequence, decision1
import copy
from Mist import decisionSeveral, decisionOnce
import time

class CompareMethods:
    def __init__(self, roll_width, given_lines, I, probab, num_period, num_sample):
        self.roll_width = roll_width  # array, mutable object
        self.given_lines = given_lines  # number, Immutable object
        self.demand_width_array = np.arange(2, 2+I)
        self.value_array = self.demand_width_array - 1
        self.I = I   # number, Immutable object
        self.probab = probab
        self.num_period = num_period   # number, Immutable object
        self.num_sample = num_sample   # number, Immutable object

    def method4(self, sequence, ini_demand, newx, roll_width):
        # newx patterns for N rows.
        newx = newx.T.tolist()

        mylist = []
        remaining_period0 = self.num_period
        sequence1 = copy.copy(sequence)
        total_usedDemand = np.zeros(self.I)
        # ini_demand1 = np.array(self.probab) * self.num_period
        deterModel = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)

        while remaining_period0:
            demand = ini_demand - total_usedDemand
            usedDemand, remaining_period = decisionSeveral(sequence, demand)
            
            diff_period = remaining_period0 - remaining_period

            demand_list = sequence[0:diff_period]

            for j in demand_list:
                for k, pattern in enumerate(newx): 
                    if pattern[j-2] > 0:
                        newx[k][j-2] -= 1
                        roll_width[k] -= j
                        break

            mylist += [1] * diff_period

            if any(usedDemand) == 0:  # all are 0
                usedDemand, decision_list = decisionOnce(
                    sequence, demand, self.probab)
                
                Indi_Demand = np.dot(usedDemand, range(self.I))

                if decision_list:
                    mylist.append(1)

                    # find the row can assign usedDemand（j)
                    for k, pattern in enumerate(newx):
                        if pattern[decision_list] > 0:
                            newx[k][decision_list] -= 1
                            if decision_list - Indi_Demand -2 >= 0: 
                                newx[k][decision_list - Indi_Demand - 2] += 1
                            roll_width[k] -= (Indi_Demand+1)
                            break

                else:
                    mylist.append(0)
                remaining_period -= 1

            remaining_period0 = remaining_period
            sequence = sequence[-remaining_period:]

            total_usedDemand += usedDemand

            # m1 = stochasticModel(self.roll_width, self.given_lines,self.demand_width_array, W, self.I, prop, dw)

            # ini_demand, newx = m1.solveBenders(eps=1e-4, maxit=20)

            # use stochastic calculate
            # demand =  deterModel.benders(roll_width, remaining_period)
            ini_demand, obj = deterModel.IP_formulation(total_usedDemand, np.zeros(self.I))

        sequence1 = [i-1 for i in sequence1 if i > 0]

        final_demand1 = np.array(sequence1) * np.array(mylist)
        final_demand1 = final_demand1[final_demand1 != 0]

        demand = np.zeros(self.I)
        for i in final_demand1:
            demand[i-1] += 1

        return demand



# mul = np.arange(2, 6)

# demand = np.array([1,2,3,4])

# demand_list = []
# for i in range(4):
#     demand_list += [demand[i]] * mul[i]


a = np.array([[1,2.0,3,4], [3,4,5,6]])
b = a.tolist()


for i in b:
    i.remove(2)
    print(i)

