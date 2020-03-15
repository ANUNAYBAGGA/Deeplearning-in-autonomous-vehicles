import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
train_data = np.load('training_data.npy',allow_pickle=True)

for i in range(len(train_data)):
    if train_data[i][1] == [0,0,0,1,0,0,0,0,0]:
        im = Image.fromarray(train_data[i][0])
        im.show()
        print(train_data[i][0])
        break

df = pd.DataFrame(train_data)
print(Counter(df[1].apply(str)))
print(len(df))
