import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000
d = 10
selected_ads = []
number_of_selection = [0] * d
sum_of_reward = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if number_of_selection[i] > 0:
            avg_reward = sum_of_reward[i]/number_of_selection[i]

            delta_i = math.sqrt((3/2) * (math.log(n + 1)) / number_of_selection[i])
            upper_bound = avg_reward + delta_i

        else:
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    selected_ads.append(ad)
    number_of_selection[ad] += 1

    reward = dataset.values[n, ad]
    sum_of_reward[ad] += reward

    total_reward += reward


plt.hist(selected_ads)
plt.title("Histogram of ads")
plt.xlabel("Ads")
plt.ylabel("Number of times ads were clicked")
plt.show()
