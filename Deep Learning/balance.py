import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter


import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import random


FILE_I_END = 7
offset = 10

try:
        random.seed()
        file_name = 'training_data.npy'
        # full file info
        train_data = np.load(file_name, allow_pickle=True)
        print('training_data.npy being balanced : ', len(train_data))
        df = pd.DataFrame(train_data)
        print(Counter(df[1].apply(str)))
        w = []
        s = []
        a = []
        d = []
        wa = []
        wd = []
        sa = []
        sd = []
        nk = []
        for data in train_data:
            img = data[0]
            choice = data[1]
            if choice == [1, 0, 0, 0, 0, 0, 0, 0, 0]:
                w.append([img, choice])
                shuffle(w)
            elif choice == [0, 1, 0, 0, 0, 0, 0, 0, 0]:
                s.append([img, choice])
                shuffle(s)
            elif choice == [0, 0, 1, 0, 0, 0, 0, 0, 0]:
                a.append([img, choice])
                shuffle(a)
            elif choice == [0, 0, 0, 1, 0, 0, 0, 0, 0]:
                d.append([img, choice])
                shuffle(d)
            elif choice == [0, 0, 0, 0, 1, 0, 0, 0, 0]:
                wa.append([img, choice])
                shuffle(wa)
            elif choice == [0, 0, 0, 0, 0, 1, 0, 0, 0]:
                wd.append([img, choice])
                shuffle(wd)
            elif choice == [0, 0, 0, 0, 0, 0, 1, 0, 0]:
                sa.append([img, choice])
                shuffle(sa)
            elif choice == [0, 0, 0, 0, 0, 0, 0, 1, 0]:
                sd.append([img, choice])
                shuffle(sd)
            elif choice == [0, 0, 0, 0, 0, 0, 0, 0, 1]:
                nk.append([img, choice])
                shuffle(nk)
            else:
                print('no matches')
        w = w[:len(s)][:len(a)][:len(d)][:len(wa)][:len(wd)][:len(nk)]
        s = s[:len(w)]
        a = a[:len(w)]
        d = d[:len(w)]
        wa = wa[:len(w)]
        wd = wd[:len(w)]
        nk = nk[:len(w)]

        final_data = w + s + a + d + wa + wd + sa + sd + nk
        shuffle(final_data)
        np.save('balanced_training_data.npy', final_data)

except Exception as e:
        print(str(e))
