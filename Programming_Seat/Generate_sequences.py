import numpy as np
from Mist import sequence_pool

# This function is used to generate and store the sequences.
count = 100
num_period = 200
# probab = [0.25, 0.3, 0.25, 0.2]
probab = [0.18, 0.7, 0.06, 0.06]
sd = 1

sequences = sequence_pool(count, num_period, probab, sd)

np.save('data_sequence' + str(probab[0]) + '.npy', sequences)
