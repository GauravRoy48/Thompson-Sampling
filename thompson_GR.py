#####################################################################################
# Creator     : Gaurav Roy
# Date        : 19 May 2019
# Description : The code performs Thompson Sampling as Reinforcement Learning
#               algorithm on the Ads_CTR_Optimisation.csv.
#####################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement Thompson Sampling
import random
N = len(dataset)
d = len(dataset.columns)

ads_Selected = []
num_Reward_1 = [0] * d
num_Reward_0 = [0] * d
total_Reward = 0

for n in range(0, N):
    max_random = 0
    ad_index = 0
    for i in range(0, d):
        random_beta = random.betavariate(num_Reward_1[i] + 1, num_Reward_0[i] + 1)
        
        if random_beta > max_random:
            max_random = random_beta
            ad_index = i
    
    ads_Selected.append(ad_index)
    reward = dataset.values[n, ad_index]
    
    if reward ==1:
        num_Reward_1[ad_index] = num_Reward_1[ad_index] + 1
    else:
        num_Reward_0[ad_index] = num_Reward_0[ad_index] + 1
        
    total_Reward = total_Reward + reward
    
# Visualize the Results
plt.hist(ads_Selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.grid(True)
plt.show()