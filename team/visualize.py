import analysis
from analysis import cats
import matplotlib.pyplot as plt
from math import log

# visualize each {x:y} in cats
cats = cats[True, True]
print(len(cats))
for key in cats:
    if log(cats[key]) > 0.0:
        plt.scatter(log(key), log(cats[key])/log(2), label=str(key))

plt.show()