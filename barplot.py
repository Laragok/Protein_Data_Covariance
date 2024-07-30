import matplotlib.pyplot as plt
import numpy as np

species = ("em_max", "ex_max", "states_0_brightness")
penguin_means = {
    'Train Set R^2 Value': (0.8479842052101813 , 0.8226919339817478 , 0.5113075760884067),
    'Test Set R^2 Value': (0,0,0),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('wavelength')
ax.set_title('')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1)

plt.show()
