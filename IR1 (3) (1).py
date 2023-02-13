#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
warnings.simplefilter("ignore")


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[5]:


zen=os.listdir("C:/Users/Sneha/Documents/IR/zen1/")
rih=os.listdir("C:/Users/Sneha/Documents/IR/rih/")
dwayne=os.listdir("C:/Users/Sneha/Documents/IR/dwayne/")


# In[6]:


limit=10
zen_images=[None]*limit
j=0
for i in zen:
    if(j<limit):
        zen_images[j]=imread("C:/Users/Sneha/Documents/IR/zen1/"+i)
        j+=1
    else:
        break
        
limit=10
rih_images=[None]*limit
j=0
for i in rih:
    if(j<limit):
        rih_images[j]=imread("C:/Users/Sneha/Documents/IR/rih/"+i)
        j+=1
    else:
        break
        
limit=10
dwayne_images=[None]*limit
j=0
for i in dwayne:
    if(j<limit):
        dwayne_images[j]=imread("C:/Users/Sneha/Documents/IR/dwayne/"+i)
        j+=1
    else:
        break


# In[7]:


imshow(zen_images[9])


# In[8]:


imshow(rih_images[0])


# In[9]:


imshow(dwayne_images[0])


# In[10]:


zen_gray=[None]*limit
j=0
for i in zen:
    if(j<limit):
        zen_gray[j]=rgb2gray(zen_images[j])
        j+=1
    else:
        break
        
rih_gray=[None]*limit
j=0
for i in rih:
    if(j<limit):
        rih_gray[j]=rgb2gray(rih_images[j])
        j+=1
    else:
        break
        
dwayne_gray=[None]*limit
j=0
for i in dwayne:
    if(j<limit):
        dwayne_gray[j]=rgb2gray(dwayne_images[j])
        j+=1
    else:
        break


# In[11]:


imshow(zen_gray[0])


# In[12]:


imshow(rih_gray[0])


# In[13]:


imshow(dwayne_gray[0])


# In[14]:


zen_gray[3].shape


# In[15]:


rih_gray[3].shape


# In[16]:


dwayne_gray[3].shape


# In[17]:


for j in range(10):
  z=zen_gray[j]
  zen_gray[j]=resize(z,(512,512))
    
for j in range(10):
  r=rih_gray[j]
  rih_gray[j]=resize(r,(512,512))
    
for j in range(10):
  d=dwayne_gray[j]
  dwayne_gray[j]=resize(d,(512,512))


# In[18]:


imshow(zen_gray[4])


# In[19]:


imshow(rih_gray[4])


# In[20]:


imshow(dwayne_gray[4])


# In[21]:


len_of_images_zen=len(zen_gray)
len_of_images_zen


# In[22]:


len_of_images_rih=len(rih_gray)
len_of_images_rih


# In[23]:


len_of_images_dwayne=len(dwayne_gray)
len_of_images_dwayne


# In[24]:


image_size_zen=zen_gray[1].shape
image_size_zen


# In[25]:


image_size_rih=rih_gray[1].shape
image_size_rih


# In[26]:


image_size_dwayne=dwayne_gray[1].shape
image_size_dwayne


# In[27]:


flatten_size_zen=image_size_zen[0]*image_size_zen[1]
flatten_size_zen


# In[28]:


flatten_size_rih=image_size_rih[0]*image_size_rih[1]
flatten_size_rih


# In[29]:


flatten_size_dwayne=image_size_dwayne[0]*image_size_dwayne[1]
flatten_size_dwayne


# In[30]:


for i in range(len_of_images_zen):
  zen_gray[i]=np.ndarray.flatten(zen_gray[i]).reshape(flatten_size_zen,1)

for i in range(len_of_images_rih):
  rih_gray[i]=np.ndarray.flatten(rih_gray[i]).reshape(flatten_size_rih,1)

for i in range(len_of_images_dwayne):
  dwayne_gray[i]=np.ndarray.flatten(dwayne_gray[i]).reshape(flatten_size_dwayne,1)


# In[31]:


zen_gray=np.dstack(zen_gray)


# In[32]:


zen_gray.shape


# In[33]:


rih_gray=np.dstack(rih_gray)
rih_gray.shape


# In[34]:


dwayne_gray=np.dstack(dwayne_gray)
dwayne_gray.shape


# In[35]:


zen_gray=np.rollaxis(zen_gray,axis=2,start=0)


# In[36]:


zen_gray.shape


# In[37]:


rih_gray=np.rollaxis(rih_gray,axis=2,start=0)
rih_gray.shape


# In[38]:


dwayne_gray=np.rollaxis(dwayne_gray,axis=2,start=0)
dwayne_gray.shape


# In[39]:


zen_gray=zen_gray.reshape(len_of_images_zen,flatten_size_zen)
zen_gray.shape


# In[40]:


rih_gray=rih_gray.reshape(len_of_images_rih,flatten_size_rih)
rih_gray.shape


# In[41]:


dwayne_gray=dwayne_gray.reshape(len_of_images_dwayne,flatten_size_dwayne)
dwayne_gray.shape


# In[42]:


zen_data=pd.DataFrame(zen_gray)
zen_gray


# In[43]:


rih_data=pd.DataFrame(rih_gray)
rih_gray


# In[44]:


dwayne_data=pd.DataFrame(dwayne_gray)
dwayne_gray


# In[45]:


zen_data["label"]="zen"
zen_data


# In[46]:


rih_data["label"]="rih"
rih_data


# In[47]:


dwayne_data["label"]="dwayne"
dwayne_data


# In[48]:


actor_1=pd.concat([zen_data,dwayne_data])


# In[49]:


actor=pd.concat([actor_1,rih_data])


# In[50]:


actor


# In[51]:


from sklearn.utils import shuffle
hollywood_indexed=shuffle(actor).reset_index()
hollywood_indexed


# In[52]:


hollywood_actors=hollywood_indexed.drop(["index"],axis=1)
hollywood_actors


# In[53]:


x=hollywood_actors.values[:,:-1]


# In[54]:


y=hollywood_actors.values[:,-1]


# In[55]:


x


# In[56]:


y


# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[59]:


from sklearn import svm


# In[60]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# In[61]:


y_pred=clf.predict(x_test)


# In[62]:


y_pred


# In[63]:


for i in (np.random.randint(0,6,4)):
  predicted_images=(np.reshape(x_test[i],(512,512)).astype(np.float64))
  plt.title("Predicted label: {0}".format(y_pred[i]))
  plt.imshow(predicted_images,interpolation="nearest",cmap="gray")
  plt.show()


# In[64]:


from sklearn import metrics


# In[65]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[66]:


accuracy


# In[67]:


from sklearn.metrics import confusion_matrix


# In[68]:


confusion_matrix(y_test,y_pred)


# In[ ]:




