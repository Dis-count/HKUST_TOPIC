import matplotlib.pyplot as plt
import numpy as np


x = [i*60 for i in range(1,5)]

primal = [1.12, 0.14, 0.26, 0.22]

BPP = [0.24, 0.04, 2.38, 8.62]

BPC = [8.62, 5.86, 3.32, 6.86]

plt.figure(figsize=(10, 6))
plt.plot(x, primal, marker='o', label='Primal')
plt.plot(x, BPP, marker='s', label='BPP')
plt.plot(x, BPC, marker='^', label='BPC')

plt.xlabel('T')
plt.ylabel('Loss')
# plt.title('Loss')
plt.legend()
# plt.grid(True, alpha = 0.3)
plt.show()
