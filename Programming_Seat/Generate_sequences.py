import numpy as np
from Mist import sequence_pool

# This function is used to generate and store the sequences.
count = 100
num_period = 100
# probab = [0.18, 0.7, 0.06, 0.06]
# probab = [0.12, 0.5, 0.13, 0.25]
probab = [0.34, 0.51, 0.07, 0.08]

sd = 1
sequences = sequence_pool(count, num_period, probab, sd)

np.save('sequence_' + str(probab[0]) + '.npy', sequences)
