import numpy as np
from collections import Counter

class samplingmethod1:
    def __init__(self, I, number_sample, number_period, prob) -> None:
        self.I = I
        self.number_sample = number_sample
        self.number_period = number_period
        # sample_multi = np.random.multinomial(self.number_period, [1/self.I]*self.I, size=self.number_sample)
        sample_multi = np.random.multinomial(
            self.number_period, prob, size = self.number_sample)
        self.sample_multi = sample_multi.tolist()

    def convert(self):
        # Converting integer list to string list
        number_arrivals = 0
        for i in self.sample_multi:
            s = [str(j) for j in i]
        # Join list items using join()
            res = " ".join(s)
            self.sample_multi[number_arrivals] = res
            number_arrivals += 1
        return self.sample_multi

    def get_prob(self):
        # Return the scenarios and the corresponding probability through sampling.
        counter_list = self.convert()
        sample_result = Counter(counter_list)
        number_scenario = len(sample_result)
        scenario_set = np.array([[0]*self.I] * number_scenario)
        count = 0
        prob_set = [0] * number_scenario
        for key, value in sample_result.items():
            key = key.split()
            int1 = [int(i) for i in key]
            prob = value / self.number_sample
            scenario_set[count] = int1
            prob_set[count] = prob
            count += 1

        return scenario_set, np.array(prob_set)


if __name__ == "__main__":
    num_sample = 5  # the number of scenarios
    I = 4  # the number of group types
    number_period = 10
    probab = [0.4, 0.2, 0.2, 0.2]
    sam = samplingmethod1(I, num_sample, number_period, probab)
    dw, prop = sam.get_prob()