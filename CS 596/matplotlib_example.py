# linePlot.py
# visualize data points and lines

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Prepare data to be plotted
np.random.seed(5) # Set seed to reproduce exactly the same figure
x = np.arange(1, 101)
y = 20 + 3*x + np.random.normal(0, 60, 100)

# Plot
ax.plot(x, y, 'o')

# Draw vertical line from (70,100) to (70, 250)
ax.plot([70, 70], [100, 250], 'k-', lw=2)

# Draw diagonal line from (70, 90) to (90, 200)
ax.plot([70, 90], [90, 200], 'k-')

# Set labels and title
ax.set(xlabel='x label', ylabel='y label',
       title='About as simple as it gets, folks')

# Save to PDF file
fig.savefig('example.pdf')
# Show
plt.show()
