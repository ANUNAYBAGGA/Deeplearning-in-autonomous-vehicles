import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from keys import PressKey,W,A,S,D,ReleaseKey

for i in range(5):
    print(5-i)
    time.sleep(1)
#PressKey(W)
#ReleaseKey(W)


def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    masked = cv2.bitwise_and(img,mask)
    return masked
def get_lane(image,m1,m2,c1,c2):
    image = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)
    y1 = 500
    y2 = int (y1*(3/5))
    #y = mx + c => x = (y-c)/m
    x1 = int((y1-c1)/m1)
    x2 = int((y2-c1)/m1)
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),15)
    x1 = int((y1-c2)/m2)
    x2 = int((y2-c2)/m2)
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),15)
    cv2.imshow("LANES",image)
    #print('y=',m1,'x + ',c1)
    #print('y=',m2,'x + ',c2)
def process(image):
    processed = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2HSV) 
    lower_yellow = np.array([18, 94, 100], dtype = "uint8")
    upper_yellow = np.array([150, 250, 255], dtype="uint8")
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(processed, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    processed = cv2.bitwise_and(processed, mask_yw)
    processed = cv2.GaussianBlur(processed,(5,5),0)
    processed = cv2.Canny(processed,threshold1=200,threshold2=300)
    #processed = cv2.GaussianBlur(processed,(5,5),0)
    vertices = np.array([[10,500] , [10,200] , [800,200] , [800,500]])
    processed = roi(processed,[vertices])
    lines = cv2.HoughLinesP(processed , 1 , np.pi/180 ,50, None,70, 5)
    m1 = 0
    m2 = 0
    c1,c2 = 1,1
    point1 = [0,0]
    point2 = [0,0]
    try:
        for x in range(len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                m = (y2-y1)/(x2-x1)
                #print(m)
                if m > 0:
                    m1+=m
                    c1+=1
                    slope = m1/(c1-1)
                    b1 = y1 - slope*x1
                elif m<0:
                    m2+=m
                    c2+=1
                    slope = m2/(c2-1)
                    b2 = y1 - slope*x1
                cv2.line(processed,(x1,y1),(x2,y2),(255,255,255),5)
        slope1,slope2 = m1/(c1-1) , m2/(c2-1)
        #y-y1 = m(x-x1)=> y - y1 = mx -mx1=> y = mx +y1-mx1, B = y1 - mx1 where B is from y = mx + b
        #b1,b2 =  y1-slope1*x1, y2 - slope2*x2
        get_lane(image,slope1,slope2,b1,b2)
    except:
        pass
    return processed,m1/c1,m2/c2
def right():
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(W)
    #PressKey(S)
    #time.sleep(0.05)
    #ReleaseKey(S)
    PressKey(D)
    PressKey(W)
    time.sleep(0.09)
    ReleaseKey(D)
    #ReleaseKey(W)

def left():
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(W)
    #PressKey(S)
    #time.sleep(0.05)
    #ReleaseKey(S)
    PressKey(A)
    PressKey(W)
    time.sleep(0.09)
    ReleaseKey(A)
    #ReleaseKey(W)
 
def none():
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)
def main():
    last_time = time.time()
    a = 1
    while (True):
        a = 2
        screenshot = ImageGrab.grab(bbox = (0,40,800,640))
        new_screen,m1,m2=process(screenshot)
        #cv2.imshow("Edges",new_screen)
        #cv2.imshow("window",cv2.cvtColor(np.array(screenshot),cv2.COLOR_BGR2RGB))
        #print("Frame took seconds = " , end = '')
        #print(time.time()-last_time)
        #print(m1,m2)
        #print(m1,m2)
        if abs(m1)-abs(m2) == 0:
            #none()
            print('reversing')
            print(m1+m2)
        if abs(m2)>abs(m1):
            #right()
            print('right')
            print(m1+m2)
        elif abs(m2) < abs(m1):
            #left()
            print("left")
            print(m1+m2)
        last_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyALLWindows()
            break
main()
