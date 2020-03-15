
import numpy as np
from grabscreen import grab_screen
import cv2
from keras.models import load_model
from getkeys import key_check
from keys import PressKey,W,A,S,D,ReleaseKey
import time

'''
w = [1,0,0,0,0,0,0,0,0]  0
a = [0,1,0,0,0,0,0,0,0]  1
s = [0,0,1,0,0,0,0,0,0]  2
d = [0,0,0,1,0,0,0,0,0]  3
wa =[0,0,0,0,1,0,0,0,0]  4
wd =[0,0,0,0,0,1,0,0,0]  5
sa =[0,0,0,0,0,0,1,0,0]  6
sd =[0,0,0,0,0,0,0,1,0]  7
nk =[0,0,0,0,0,0,0,0,1]  8
'''
for i in range(5):
    print(5-i)
    time.sleep(1)

def straight():
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    PressKey(W)
def right():
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(W)
    PressKey(D)
def left():
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(W)
    PressKey(A)
def reverse():
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    PressKey(S)
def straight_left():
    ReleaseKey(D)
    ReleaseKey(S)
    PressKey(W)
    PressKey(A)
def straight_right():
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(W)
    PressKey(D)
def reverse_left():
    ReleaseKey(D)
    ReleaseKey(W)
    PressKey(S)
    PressKey(A)
def reverse_right():
    ReleaseKey(A)
    ReleaseKey(W)
    PressKey(S)
    PressKey(D)
def no_key():
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)



paused = False
a = 1
print("LOADING MODEL ....... ")
model = load_model("xception_model_gtaV.h5")
#model.summary()
print("MODEL LOADED ........ ")
#weights = np.array([0.030903154382632643, 0.020275559590445278, 1000.0, 0.013302794647291147, 0.0355283995449392, 0.025031555049932444, 1000.0, 1000.0, 0.016423203268260675])

while True:
    a = 2
    if not paused:
        screen = grab_screen(region=(0,40,800,640))
        screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB)
        screen = cv2.resize(screen, (160,120))
        screen = np.array([screen])
        y = model.predict(screen)
        #y *= weights
        mode_choice = np.argmax(y)
        if mode_choice == 0:
            straight()
            choice = "straight"
        elif mode_choice == 1:
            left()
            choice = "left"
        elif mode_choice == 2:
            reverse()
            choice = "reverse"
        elif mode_choice == 3:
            right()
            choice = "right"
        elif mode_choice == 4:
            straight_left()
            choice = "straight left"
        elif mode_choice == 5:
            straight_right()
            choice = "straight right"
        elif mode_choice == 6:
            reverse_left()
            choice = "reverse left"
        elif mode_choice == 7:
            reverse_right()
            choice = "reverse_right"
        else:
            no_key()
            choice = "no key"
        print(choice)
        keys = key_check()
    if 'T' in keys:
        no_key()
        print('Pausing!')
        paused = True
        time.sleep(15)
        print("Unpaused")
        paused = False
        no_key()
