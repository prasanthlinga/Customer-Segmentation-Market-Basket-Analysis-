#!/usr/bin/env python
# coding: utf-8

# In[1099]:


import numpy as np
import pandas as pd


# In[1100]:


data=pd.read_csv(r"C:\Users\prasa\OneDrive\Desktop\Mall_Customers.csv")  # raw strings 
# backslash is treated as an escape character  if we dont use r , 
data


# In[1101]:


data.describe(include="all")  # NaN stands for not a number


# In[1102]:


data.info()


# In[1103]:


data["Gender"].value_counts()


# In[1104]:


#data["Gender"]=1
data


# In[1105]:


#if data.Gender=="Male":
 #   data["Gender"]=1  
#else:
#    data["Gender"]=0
#data


# In[ ]:





# In[1106]:


#dummy_data=pd.get_dummies(data.Gender,prefix="Gender")
#dummy_data


# In[1107]:


#data=data.drop('Gender',axis=1)
data["Gender"].replace({"Female":0,"Male":1},inplace=True)
data


# In[1108]:


#data=pd.concat([data,dummy_data],axis=1)
data


# In[1109]:


data.isnull()


# In[1110]:


data.isnull().values.any()


# In[1111]:


data.isnull().sum()


# In[1112]:


# we can drop the customer id column 
data=data.drop(["CustomerID"],axis=1)


# In[1113]:


data


# In[1114]:


#from sklearn.preprocessing import MinMaxScaler
#scaler=MinMaxScaler()
#transformed_data=scaler.fit_transform(data)
#from sklearn.preprocessing import StandardScaler
#scaler=StandardScaler()
#transformed_data=scaler.fit_transform(data)


# # We are getting good results without scaling the data

# In[1115]:


transformed_data=data


# In[1116]:


new_data=pd.DataFrame(transformed_data)
new_data


# In[1117]:


new_data.columns=data.columns


# In[1118]:


new_data


# In[1119]:


new_data.corr()


# In[1120]:


import seaborn as sns
sns.pairplot(new_data)


# In[1121]:


sns.heatmap(new_data.corr(),annot=True,cmap="cubehelix",fmt="0.2f")


# # K Means  Clustering
# 

# In[1122]:


from sklearn.cluster import KMeans
WCSS=[] # within the cluster sum of squares
for i in range(1,20):
    # 300 iterations for single run and n_init=10 is no.of times kmeans algo will run on different centroid seeds
    kmeans=KMeans(n_clusters=i,random_state=42,init="k-means++",max_iter=300,n_init=10)#see o/p default clsters=8 check sklearn
    print(kmeans)
    kmeans.fit(new_data)
    WCSS.append(kmeans.inertia_)
    print(WCSS)
    

    
    


# In[1123]:


print(WCSS)


# In[1124]:


import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot(WCSS)


# In[1125]:


kmeans=KMeans(n_clusters=5,random_state=42,max_iter=300,n_init=10)
pred_y=kmeans.fit_predict(new_data)
pred_y


# In[1126]:


print(kmeans.labels_)
print(kmeans.n_iter_)


# In[1127]:


print(kmeans.cluster_centers_)


# In[1128]:


plt.figure(figsize=(10,8))
plt.scatter(new_data.iloc[ :,2],new_data.iloc[ :,3])  # annual income vs spending score


# In[1129]:


plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],c="red")


# In[1130]:


plt.figure(figsize=(10,8))
plt.scatter(new_data.iloc[ :,2],new_data.iloc[ :,3])
plt.scatter(kmeans.cluster_centers_[:,2],kmeans.cluster_centers_[:,3],c="red")


# # DBSCAN Clustering

# In[1131]:


from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=12,min_samples=4,metric="euclidean")#min samples default=5, eps default=0.5
dbscan.fit(new_data)
y_pred=dbscan.fit_predict(new_data)
y_pred       # fit_predict calls fit method and returns labels..not a difference b/w fit and fit_predict


# In[1132]:



print(dbscan.labels_)


# In[1133]:


print(dbscan.core_sample_indices_)


# In[1134]:


print(dbscan.components_)


# In[1135]:


n_clusters=len(set(dbscan.labels_))- (1 if -1 in dbscan.labels_ else 0)
n_clusters


# In[1136]:


print(metrics.silhouette_score(new_data,dbscan.labels_))


# In[1137]:


plt.scatter(new_data.iloc[:,2],new_data.iloc[:,3],c=y_pred)


# In[ ]:





# In[ ]:




