#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df=pd.read_csv("D:/CHRIST UNIVERSITY/DALMP/dalmp1.csv")


# In[3]:


###Loading the dataset


# In[4]:


df.head()


# #finding stats about the data set mainly number of null values. There are total of 908 values in the dataset and only coordinates have less than 908 values implying that coordinates have null values

# In[5]:


df.info()


# In[6]:


dupli=df.duplicated()


# In[7]:


print(dupli)


# In[8]:


missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[9]:


df=df.dropna()


# In[10]:


#we are droping null values because we are doing spatial analysis and without coordinates we can use the data


# In[11]:


df.info()


# In[12]:


missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[13]:


dupli=df.duplicated("District")


# In[14]:


#checking duplicate values in districts 


# In[15]:


print(dupli)


# In[16]:


# removing the second pair of  duplicate value  


# In[17]:


df = df.drop_duplicates(subset=['District'], keep='first')
print(df)


# In[18]:


# this is basic information of the dataset after cleaning


# In[19]:


df.info()


# In[20]:


statistics = df.describe()
print("Descriptive Statistics:\n", statistics)


# In[21]:


df.head(25)


# In[22]:


df['Total A'] = df['A.Murder (Sec.302 IPC)'] +df['A. Culpable Homicide not amounting to Murder (Sec.304 IPC)'] + df['A. Causing Death by Negligence'] + df['A. Hurt'] + df['A. Assault on Women with Intent to Outrage her Modesty'] + df['A. Kidnapping and Abduction']


# In[23]:


df


# In[24]:


df['Total D'] = df['D. Theft (Section 379 IPC)'] + df["D. Burglary (Sec.454 to 460 r/w Sec.380 IPC)"]+ df["D. Dacoity"]+df["D. Others"]


# In[25]:


df['Total E']= df['E. Counterfeiting']+ df['E. Forgery, Cheating & Fraud']+df['E. Offences Relating to Documents & Property Marks (Total)']


# In[26]:


df['Total F']= df['F. Offences relating to Elections (Sec.171-E to 171-I r/w IPC/SLL)']+ df['F. Disobedience to order duly promulgated by Public Servant (Sec.188)']+df['F. Harbouring an Offender (Sec.212/216/216A IPC)']+df['F. Offences relating to Adulteration or Sale of Food/Drugs (Sec.272/273/274/275/276 IPC)']+df['F. Rash Driving on Public way']+df['F. Others']


# In[27]:


# We are adding all the similar crime so that we can show that together in sptaial analysis and also find percentage of each type of crime in each state


# In[28]:


# Standardization
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(scaler_standard.fit_transform(df[['Total A', 'Total D','Total E','Total F']]), columns=['Total A1', 'Total D','Total E','Total F'])


# In[29]:


# Display the standardized dataset
print("\nStandardized Dataset:")
print(df_standardized)


# In[30]:


# Display summary statistics
summary_statistics = df_standardized.describe()


# In[31]:


# Print the summary statistics of standardized features
print("Summary Statistics of Standardized Features:\n", summary_statistics)


# In[32]:


plt.figure(figsize=(10, 6))
sns.histplot(df["Total A"], bins=20, kde=True)
plt.title("Distribution of Total body crime rates in India(Table A)")
plt.show()


# In[33]:


plt.figure(figsize=(10, 6))
sns.histplot(df_standardized["Total A1"], bins=20, kde=True)
plt.title("Distribution of standarised vales of table A")
plt.show()


# In[34]:


# Min-Max Scaling
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df[['Total A', 'Total D','Total E','Total F']]), columns=['Total A1.1', 'Total D','Total E','Total F'])


# In[35]:


# Display the Min-Max scaled dataset
print("\nMin-Max Scaled Dataset:")
print(df_minmax)


# In[36]:


plt.figure(figsize=(10, 6))
sns.histplot(df_minmax ["Total A1.1"], bins=20, kde=True)
plt.title("Distribution of Min-Max values of Table A")
plt.show()


# In[37]:


# Display summary statistics
summary_statistics_minmax = df_minmax.describe()


# In[38]:


print("Summary Statistics of Min-Max Scaled Features:\n", summary_statistics_minmax)


# In[39]:


plt.figure(figsize=(10,5))
sns.histplot(df["Total Cognizable IPC crimes"], bins=20, kde=True)
plt.title("Distribution of Total  crime rates in India(Table A)")
plt.show()


# In[ ]:





# In[40]:


# Scatter Plot
x = df['A. Causing Death by Negligence']
y = df['F. Rash Driving on Public way']
plt.figure(figsize=(20, 10))
plt.scatter(x, y, label='Scatter Plott', color='red', marker='o')
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[41]:


# Scatter Plot
x = df['E. Forgery, Cheating & Fraud']
y = df['E. Offences Relating to Documents & Property Marks (Total)']
plt.figure(figsize=(8, 4))
plt.scatter(x, y, label='Scatter Plott', color='red', marker='o')
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[ ]:





# In[42]:





# Sample data
x = df['State/UT']
y = df['Total Cognizable IPC crimes']

# Create a colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(x)))

# Create the bar chart with different colors for each state
plt.figure(figsize=(20, 10))
bars = plt.bar(x, y, color=colors, alpha=0.8)

# Customize the plot
plt.title('Bar Chart')
plt.xlabel('State/UT')
plt.ylabel('Total Cognizable IPC crimes')

# Show a color bar to indicate the mapping of colors to states
color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=plt.gca(), pad=0.2)
color_bar.set_label('Color Intensity')

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()

# Show the plot
plt.show()


# In[43]:


import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
# Assuming df is your DataFrame

# Set the style of seaborn
sns.set(style="whitegrid")

# Pair Plot without specifying markers
sns.pairplot(df, hue="State/UT")
plt.title("Pair Plot of Your Dataset")
plt.show()


# In[44]:


df


# In[56]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with the specified columns
# Example: df = pd.read_csv('your_data.csv')

# Extract relevant columns for visualization
crime_types = df.columns[4:-5]  # Select columns from 'A.Murder' to 'F. Miscellaneous'

# Sum the counts of each crime type across all states/UT
crime_counts = df[crime_types].sum()

# Plotting a pie chart
plt.figure(figsize=(16, 12))
plt.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Crime Types')
plt.show()


# In[46]:


# Assuming df is your DataFrame with the specified columns
# Example: df = pd.read_csv('your_data.csv')

# Plotting a boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Total Cognizable IPC crimes', data=df, palette='pastel')
plt.title('Boxplot for Total Cognizable IPC Crimes')
plt.xlabel('Total Cognizable IPC Crimes')
plt.show()

