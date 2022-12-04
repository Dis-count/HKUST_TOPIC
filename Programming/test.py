import numpy as np
from Method4 import deterministicModel

class FirstComeFirstServe:
    def __init__(self, I, num_period, roll_width, given_lines) -> None:
        self.I = I
        self.num_period = num_period
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.num_seats = roll_width * given_lines

    def binary_search_first(self, sequence):
        # Return the index not less than the first 
        sequence = [i-1 for i in sequence]
        target = self.num_seats
        arr = np.cumsum(sequence) + np.arange(1, self.num_period+1)
        low = 0
        high = len(arr)-1
        res = -1
        while low <= high:
            mid = (low + high)//2
            if target <= arr[mid]:
                res = mid
                high = mid-1
            else:
                low = mid+1
        if res == -1:
            total = sum(sequence)
            seq = sequence
        else:
            total = sum(sequence[0:res])

        remaining = target - total
        if remaining > 0 and res >0:
            for i in sequence[res:]:
                if i == remaining-1:
                    seq = sequence[0:res]+ [i]
        return seq

    def seq2demand(self):
        seq = self.binary_search_first()
        demand = np.zeros()
        for x in seq:
            demand[x-1] += 1
        return demand

    def row_by_row(self, sequence):
        # i is the i-th request in the sequence
        # j is the j-th row
        # sequence includes social distance.
        remaining_capacity = np.zeros(self.given_lines)
        current_capacity = self.roll_width
        j = 0
        period  = 0
        for i in sequence:
            if i in remaining_capacity:
                inx = np.where(remaining_capacity == i)[0][0]
                remaining_capacity[inx] = 0

            if current_capacity[j] > i:
                current_capacity[j] -= i
            else:    
                remaining_capacity[j] = current_capacity[j]
                j +=1
                if j > self.given_lines-1:
                    break
                current_capacity[j] -= i
            period +=1
        
        lis = [0] * (self.num_period - period)
        for k, i in enumerate(sequence[period:]):
            if i in remaining_capacity:
                inx = np.where(remaining_capacity == i)[0][0]
                remaining_capacity[inx] = 0
                lis[k] = 1
        my_list = [1]* period + lis
        sequence = [i-1 for i in sequence]

        final_demand = np.array(sequence) * np.array(my_list)
        final_demand = final_demand[final_demand != 0]

        demand = np.zeros(self.I)
        for i in final_demand:
            demand[i-1] += 1

        return demand

    def offline(self, sequence):
        # This function is to obtain the optimal decision.
        demand = np.zeros(self.I)
        sequence = [i-1 for i in sequence]
        for i in sequence:
            demand[i-1] += 1
        test = deterministicModel(
            self.roll_width, self.given_lines, self.demand_width_array, self.I)
        newd, obj = test.IP_formulation(np.zeros(self.I), demand)

        return newd
