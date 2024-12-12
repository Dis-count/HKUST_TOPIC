import numpy as np
import matplotlib.pyplot as plt

# This function is used to load the data and plot the corrsponding graphs.
# The specific parameters are as follows:

def plot_data(given_lines, roll_width, data):
    sd = 1    
    t_value = data[0]
    people_value = data[1]
    occup_1 = data[2]
    occup_2 = data[3]

    total_seat = np.sum(roll_width) - given_lines * sd

    plt.plot(t_value, people_value, 'b-', label= 'Without social distancing')
    plt.plot(t_value, occup_1, 'r--', label='With 1 social distancing')
    plt.plot(t_value, occup_2, 'y*', label='With 2 social distancing')

    plt.xlabel('Periods')
    plt.ylabel('Percentage of total seats')

    plt.legend()
    plt.savefig('distance.pdf')

given_lines = 10
roll_width = np.ones(given_lines) * 21
data = np.load('data_distances.npy')
plot_data(given_lines, roll_width, data)
