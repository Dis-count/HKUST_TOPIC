import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plot Results of Different Policies under multiple probabilities


df = pd.read_excel('plot.xlsx')

x = range(1, len(df) + 1)
y_columns = df.iloc[:, 1:5]

plt.figure(figsize=(10, 6))

colors = ['blue', 'red', 'green', 'orange']
line_styles = ['-', '--', '-.', ':']


labels = ['SPBA', 'RDPH', 'BPC', 'BLC']

for i, col in enumerate(y_columns.columns):
    plt.plot(x, df[col],
             color = colors[i],
             linestyle=line_styles[i],
             markersize=6,
             linewidth=2,
             label=labels[i])


plt.xlabel('Probability Distribution Index', fontsize=12)
plt.ylabel('Ratio (%)', fontsize=12)
plt.title('T = 80', fontsize=12)
plt.legend(loc='lower right')

plt.show()
