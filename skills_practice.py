#%%
import pandas as pd
import sklearn as sk
import numpy as np
# %%
salary_data = pd.read_csv('2025_salaries.csv', header=1)
salary_data.head()
# %%
stats = pd.read_csv('nba_2025.txt', sep=',', encoding='latin-1')
stats.head()
# %%
# help(pd.merge)
# %%
merged_data = pd.merge(salary_data, stats, on='Player')
merged_data.head()

# %%
duplicates = merged_data[merged_data.duplicated(subset='Player',keep=False)]
duplicates.head()
# %%
# tip for clustering: experiment with plotting 2d scatterplots to see which
# features show the clusters the best, then use that plot and color it by salary
# look for high performers, low salary
# shape shows cluster: low performers, mid performers, high performers
# color shows salary

# select variables with high variance, helps you separate players much easier