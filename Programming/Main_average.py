import Main
import numpy as np

group_type, final_demand1, final_demand3, final_demand4 = Main.result()

final_demand1 = np.zeros(group_type+1)
final_demand3 = np.zeros(group_type+1)
final_demand4 = np.zeros(group_type+1)

for i in range(1):
    z0, a,b,c = Main.result()

    final_demand1 += a
    final_demand3 += b
    final_demand4 += c


# print(final_demand1/50)
# print(final_demand3/50)
# print(final_demand4/50)
