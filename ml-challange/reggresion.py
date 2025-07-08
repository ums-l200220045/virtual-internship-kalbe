#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# ## Import Data

# In[2]:


df_customer = pd.read_csv('Case Study - Customer.csv', sep= ';')
df_product = pd.read_csv('Case Study - Product.csv', sep= ';')
df_store = pd.read_csv('Case Study - Store.csv', sep= ';')
df_transaction = pd.read_csv('Case Study - Transaction.csv', sep= ';')


# In[3]:


print(df_customer)


# In[4]:


print(df_product)


# In[5]:


print(df_store)


# In[6]:


print(df_transaction)


# ## Data Cleansing & Preparation

# In[7]:


# Mengecek nilai-nilai yang hilang
missing_values1 = df_customer.isnull().sum()
missing_values2 = df_product.isnull().sum()
missing_values3 = df_store.isnull().sum()
missing_values4 = df_transaction.isnull().sum()

# Mengecek duplikat
duplicate_rows1 = df_customer.duplicated()
duplicate_rows2 = df_product.duplicated()
duplicate_rows3 = df_store.duplicated()
duplicate_rows4 = df_transaction.duplicated()

# Menampilkan statistik deskriptif untuk melihat ketidaksesuaian data
df_customer.describe()


# In[8]:


df_customer.info()


# In[9]:


df_product.describe()


# In[10]:


df_product.info()


# In[11]:


df_store.describe()


# In[12]:


df_store.info()


# In[13]:


df_transaction.describe()


# In[14]:


df_transaction.info()


# ## Menggabungkan Data

# In[15]:


merged_data1 = pd.merge(df_customer, df_transaction, on='CustomerID')
merged_data2 = pd.merge(merged_data1, df_store, on='StoreID')
merged = pd.merge(merged_data2, df_product, on='ProductID')
merged.head()


# In[16]:


merged.info()


# In[17]:


merged.duplicated().sum()


# ## Membuat Model Regresi (Time Series)

# In[18]:


# Mengubah tipe data tanggal ke format datetime
merged['Date'] = pd.to_datetime(merged['Date'])
merged['Longitude'] = merged['Longitude'].apply(lambda x: x.replace(',','.')).astype(float)
merged['Latitude'] = merged['Latitude'].apply(lambda x: x.replace(',','.')).astype(float)


# In[19]:


# Membuat Data Time Series
daily_data = merged.groupby('Date')['Qty'].sum().reset_index()


# In[20]:


# Persiapan Data Time Series
data = daily_data.set_index('Date')
data2 = data.resample('D').sum()  # Resampling harian


# In[21]:


data2.head()


# In[22]:


# Visualize Data
data2.plot(figsize=(12,6))


# In[23]:


# Memisahkan Data
train_size = int(len(data2) * 0.8)
train_data, test_data = data2[:train_size], data2[train_size:]
print(train_data.shape, test_data.shape)


# In[24]:


import seaborn as sns
plt.figure(figsize=(12,5))
sns.lineplot(data=train_data, x=train_data.index, y=train_data['Qty'])
sns.lineplot(data=test_data, x=test_data.index, y=test_data['Qty'])
plt.show()


# In[25]:


from statsmodels.tsa.arima.model import ARIMA

# Misalkan Anda sudah memiliki data time series pelatihan dalam 'train_data'

# Langkah 1: Tentukan nilai p, d, dan q
p = 2  # Order of Autoregression
d = 2  # Degree of Differencing
q = 2  # Order of Moving Average

# Langkah 2: Buat model ARIMA dengan parameter yang telah ditentukan
model = ARIMA(train_data, order=(p, d, q))

# Langkah 3: Latih model menggunakan data pelatihan
model_fit = model.fit()


# In[26]:


start_idx = len(train_data)
end_idx = len(train_data) + len(test_data) - 1
predictions = model_fit.predict(start=start_idx, end=end_idx, dynamic=False)

# Evaluasi performa
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_data, predictions)
print(f"Mean Squared Error: {mse}")


# In[27]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(test_data, label='Qty')
plt.plot(predictions, color='red', label='Predicted')
plt.legend()
plt.show()