import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

w = [1,0,0,0,0,0,0,0,0]
a = [0,1,0,0,0,0,0,0,0]
s = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa =[0,0,0,0,1,0,0,0,0]
wd =[0,0,0,0,0,1,0,0,0]
sa =[0,0,0,0,0,0,1,0,0]
sd =[0,0,0,0,0,0,0,1,0]
nk =[0,0,0,0,0,0,0,0,1]



numpy_data = np.load('training_data_without_balancing.npy',allow_pickle=True)
df = pd.DataFrame(data=numpy_data, columns=["Images", "Keys"])

for i in range(len(df)):
    if df["Keys"][i] == nk:
        df["Keys"][i] = "nk"
    elif df["Keys"][i] == w:
        df["Keys"][i] = "w"
    elif df["Keys"][i] == a:
        df["Keys"][i] = "a"
    elif df["Keys"][i] == s:
        df["Keys"][i] = "s"
    elif df["Keys"][i] == d:
        df["Keys"][i] = "d"
    elif df["Keys"][i] == wa:
        df["Keys"][i] = "wa"
    elif df["Keys"][i] == wd:
        df["Keys"][i] = "wd"
        
print("DataSet summary : ")
print(df.head())
print()
print("Total Data : ",len(df))
print()
print("Key Distribution : ")
print(df["Keys"].value_counts())
ax = df['Keys'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Keys Distribution")
ax.set_xlabel("Keys")
ax.set_ylabel("Frequency")
plt.show()
print()


#Merge WA+A , WD+D and S+NK as they are similar
for i in range(len(df)):
    if df["Keys"][i] == "nk":
        df["Keys"][i] = "stop"
    elif df["Keys"][i] == "w":
        df["Keys"][i] = "forward"
    elif df["Keys"][i] == 'a':
        df["Keys"][i] = "left"
    elif df["Keys"][i] == 's':
        df["Keys"][i] = "stop"
    elif df["Keys"][i] == 'd':
        df["Keys"][i] = "right"
    elif df["Keys"][i] == 'wa':
        df["Keys"][i] = "left"
    elif df["Keys"][i] == 'wd':
        df["Keys"][i] = "right"
    
print("Key Distribution : ")
print(df["Keys"].value_counts())
ax = df['Keys'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Keys Distribution")
ax.set_xlabel("Keys")
ax.set_ylabel("Frequency")