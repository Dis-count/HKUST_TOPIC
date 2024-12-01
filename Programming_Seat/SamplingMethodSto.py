import numpy as np
from collections import Counter

# The funtion is used when compare the stochastic values.

class samplingmethod1:
    def __init__(self, I: int, number_sample, number_period: int, prob, sd: int) -> None:
        self.I = I
        self.number_sample = number_sample
        self.number_period = number_period
        self.prob = prob
        self.s = sd
        sample_multi = np.random.multinomial(self.number_period, prob, size = self.number_sample)
        self.sample_multi = sample_multi.tolist()

    def accept_sample(self, seq):
        sample = np.array(self.sample_multi)
        sample[:, seq-1-self.s] = sample[:, seq-1-self.s] + 1
        self.sample_acc = sample.tolist()
        counter_list = self.convert_acc(self.sample_acc)
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

    def convert_acc(self, sample):
        # Converting integer list to string list
        number_arrivals = 0
        for i in sample:
            s = [str(j) for j in i]
        # Join list items using join()
            res = " ".join(s)
            sample[number_arrivals] = res
            number_arrivals += 1
        return sample

    def convert_ini(self, seq):
        # Converting integer list to string list
        sample_multi_ini = np.random.multinomial(self.number_period, self.prob, size=self.number_sample)
        sample_multi_ini[:, seq-1-self.s] = sample_multi_ini[:, seq-1-self.s] + 1
        sample_multi_ini = sample_multi_ini.tolist()
        number_arrivals = 0
        for i in sample_multi_ini:
            s = [str(j) for j in i]
        # Join list items using join()
            res = " ".join(s)
            sample_multi_ini[number_arrivals] = res
            number_arrivals += 1
        return sample_multi_ini

    def get_prob_ini(self, seq):
        # Return the scenarios and the corresponding probability through sampling.
        counter_list = self.convert_ini(seq)
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
    sd = 2
    probab = [0.4, 0.2, 0.2, 0.2]
    sam = samplingmethod1(I, num_sample, number_period, probab, sd)
    sam.accept_sample(2)
    dw, prop = sam.get_prob()
    print(dw)
