import numpy as np
from SamplingMethod import samplingmethod
from Mist import generate_sequence, several_class

# This function uses dynamic way to solve the problem directly.

class dynamicWay:
    def __init__(self, roll_width, given_lines, I, prop):
        self.roll_width = roll_width
        self.given_lines = given_lines
        self.demand_width_array = np.arange(2, 2+I)
        self.value_array = self.demand_width_array - 1
        self.I = I
        self.prop = prop

    def greedyLargest(self):
        # Generate the largest pattern for each row.
        patterns = np.zeros((self.given_lines, self.I+2))
        for i in range(self.given_lines):
            seat_width = self.roll_width[i]

            lar_pattern = []

            while seat_width >= self.I+1:
                seat_width -= self.I+1
                lar_pattern.append(self.I+1)
            lar_pattern.append(int(seat_width))

            demand = np.zeros(self.I+2)
            for j in lar_pattern:
                demand[j] += 1
            patterns[i] = demand

        return patterns

    def largest(self, sequence):
        #  For each arrival, use the largest pattern to accept
        patterns = self.greedyLargest()
        which_row = []
        accept_list = []
        remaining_period = len(sequence)

        for arrival in sequence:
            cnt_2 = 0
            for i in range(self.given_lines):
                pattern = patterns[i]
                if sum(pattern) < 2:
                    cnt_2 += 1
                    continue
                mul = np.arange(1, self.I +3)
                pat = [1 if x > 0 else 0 for x in pattern]
                mul = mul * pat
                mul = mul[mul > 0]
                remain_seat = int(min(mul)-1)
                max_seat = int(max(mul)-1)

                if pattern[arrival] > 0:
                    pattern[arrival] -= 1
                    which_row.append(i)
                    accept_list.append(1)
                    break

                if remain_seat + max_seat >= arrival and arrival > remain_seat:  # Accept
                    patterns[i][max_seat] -= 1
                    patterns[i][remain_seat] -= 1
                    obj_seat = max_seat + remain_seat - arrival
                    patterns[i][obj_seat] += 1
                    which_row.append(i)
                    accept_list.append(1)
                    break

                cnt_2 += 1

            if cnt_2 == self.given_lines:
                sum_supply = np.sum(patterns, 0)  # sum by column, delete the first two results.

                accept_reject = several_class(arrival-1, sum_supply[2:], remaining_period-1, self.prop)

                if accept_reject:
                    sum_supply[accept_reject + 2] -= 1
                    sum_supply[accept_reject + 2 - arrival] +=1

                    for j in range(self.given_lines):
                        pattern = patterns[j]
                        if pattern[accept_reject+2] > 0:
                            pattern[accept_reject+2] -= 1
                            pattern[accept_reject+2 - arrival] += 1
                            which_row.append(j)
                            accept_list.append(1)
                            break
                else:
                    accept_list.append(0)

            remaining_period -= 1
        seq = np.array(sequence) -1
        acc_people = np.dot(accept_list, seq)

        return acc_people


if __name__ == "__main__":
    num_sample = 1000  # the number of scenarios
    I = 4  # the number of group types
    number_period = 60
    given_lines = 10
    np.random.seed(0)

    probab = [0.4, 0.4, 0.1, 0.1]
    sam = samplingmethod(I, num_sample, number_period, probab)

    roll_width = np.arange(21, 21 + given_lines)
    total_seat = np.sum(roll_width)

    sequence = generate_sequence(number_period, probab)

    test = dynamicWay(roll_width, given_lines, I, probab)

    pat = test.greedyLargest()

    acc_people = test.largest(sequence)

    print(acc_people)

