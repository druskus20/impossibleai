import cv2
import time

import numpy as np

keypresses = np.load("keypresses_nodeaths.npz")['arr_0']
cap = cv2.VideoCapture("video_nodeaths.mp4")
frames = np.load('processed_data.npz',  allow_pickle=True)['arr_0']
print(len(frames))
print(len(keypresses))

ret = True
i = 0
while (cap.isOpened() and ret):
    ret, frame = cap.read()
    #frame = frames[i]
    cv2.imshow('frame',frame)
    cv2.waitKey(1) 
    print(keypresses[i])
    time.sleep(0.01)
    i+=1







