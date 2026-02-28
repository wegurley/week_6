# %%
# load libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# %%
# load data
df = pd.read_csv('house_votes_Dem.csv', encoding='latin-1')
# help(pd.read_csv)
# %%
# take a look at the data
df.head()
df.info()
# %%
# separate out the numeric features
c_num = df[['aye','nay','other']]
# don't need to standardize this data, it's on the same scale already
# %%
# documentation for kmeans in sklearn
help(KMeans)
# defaults initation of clusters to k-means++. 
# %% build a kmeans model
kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
#verbose=1 turns on verbosity mode.
kmeans.fit(c_num)
# %% look at the information in the model
print(kmeans.cluster_centers_)
print(kmeans.labels_)
# %%
# add the cluster labels to the original data frame
df['cluster'] = kmeans.labels_
df.head()
# %%
  
# %% simple plot of the clusters
plt.scatter(data=df,x='aye',y='nay')
 
# %%
# use a for loop to check different cluster
# numbers and see how the inertia changes
inertias=[]
k_values = range(1,10)
for k in k_values:
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(c_num)
    inertias.append(kmeans.inertia_)

plt.plot(k_values,inertias,marker='o')


# %%
