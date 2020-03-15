import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

w = [1,0,0,0,0,0,0,0,0]
a = [0,1,0,0,0,0,0,0,0]
s = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa =[0,0,0,0,1,0,0,0,0]
wd =[0,0,0,0,0,1,0,0,0]
sa =[0,0,0,0,0,0,1,0,0]
sd =[0,0,0,0,0,0,0,1,0]
nk =[0,0,0,0,0,0,0,0,1]

def keys_to_output(keys):
    output = []
    if 'A' in keys and 'W' in keys:
        output = wa
    elif 'D' in keys and 'W' in keys:
        output = wd
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    elif 'W' in keys:
        output = w
    elif 'D' in keys and 'S' in keys:
        output = sd
    elif 'A' in keys and 'S' in keys:
        output = sa
    elif 'S' in keys:
        output = s
    else:
        output = nk
    return output
file_name = 'training_data.npy'
if os.path.isfile(file_name):
    print('Loading dataset....')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, making new file called training_data.npy .....')
    training_data = []
def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    paused = False
    cnt = 0
    while(True):
        if not paused:
            screen = grab_screen(region=(0,40,800,640))
            last_time = time.time()
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (160,120))
            keys = key_check()
            output = keys_to_output(keys)
            cv2.imshow("grayscreen",screen)
            training_data.append([screen,output])
            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name,training_data)
                print("Chal raha hai, drive dhungh se kar")
            cnt+=1
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

    return cnt
cnt = main()
print(cnt)
