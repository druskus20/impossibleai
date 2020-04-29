from grabber import Grabber
import numpy as np
import cv2
import time
import signal
import sys
import win32api as wapi
from scipy.linalg import norm

# This class will contain all the screen capture functionality
class ScreenRecorder(object):
    grp = None
    dimmensions = None # Screen area

    attempt = 1
    keypresses = []
    buffer_write = None     # Write pointer
    buffer_read = None      # Read pointer
    space_pressed = False

    attempt_startime = 0
    attempt_endtime = 0

    times = []              # List with the time progression of each attempt

    writter = None      #  For saving video

    # sfrag example: (1, 26, 1601, 926) captures 1600x900 
    # without the window bar 
    def __init__(self, sfrag, buffer_len=2):
        self.grb = Grabber(bbox=sfrag)

        # Calculate screen size
        size = (sfrag[2]-sfrag[0], sfrag[3]-sfrag[1])
        self.dimmensions = size + (3,)

        # Set the frame buffer to zeros
        self.buffer_write = np.zeros(self.dimmensions, dtype=np.int8)
        self.buffer_read = np.zeros(self.dimmensions, dtype=np.int8)

        self.space_pressed = False   

   

    # Grabs a frame and sotores it in buffer[buffer_head]
    # Also grabs the key value
    def refresh_frame(self):
        if wapi.GetAsyncKeyState(32):
            self.space_pressed = True
        else:
            self.space_pressed = False
            
        self.buffer_write = self.grb.grab(None)
        self.buffer_write, self.buffer_read = self.buffer_read, self.buffer_write
        
 

    # Gets the newest frame from buffer[buffer_head]
    def get_newest_frame(self):
        return self.buffer_read

    def capture_live(self, show=False, save=False, savePath="video_raw.mp4", time_frame=5, cut_attempt=False):
      
        time_start = time.time()
        if save and (savePath != None): 
            fps=60 # !!!
            
            if cut_attempt:
                size = (self.dimmensions[0]-200, self.dimmensions[1]-100)
            else:
                size = (self.dimmensions[0], self.dimmensions[1])

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(savePath, fourcc, fps, size)  

        self.attempt_startime = time.time()
        while True:
            
            # Cambiar por tecla !!!
            if time.time() - time_start > time_frame:
                break

            self.refresh_frame()
            self.keypresses.append(self.key_check(32))
            self.update_attempt()
                    
            # Recortar el frame
            if cut_attempt:
                frame = self.reduce_frame(self.buffer_read)
            else:
                frame = self.buffer_read

            # Muestra la grabacion
            if show:
                #frame = self.buffer_read
                cv2.imshow('frame',frame)
                cv2.waitKey(1) 

            if save:
                #frame = self.buffer_read
                time.sleep(0.01)
                writer.write(frame)
    
        if save: 
            writer.release()
            np.asarray(self.keypresses)
            np.savez_compressed("keypresses_raw.npz", self.keypresses, allow_pickle=True)

    def update_attempt(self, tolerance=0):
        last_frame = self.get_attempt_area(self.buffer_read)
        last_last_frame = self.get_attempt_area(self.buffer_write)
        
        if self.attempt_has_changed(last_frame, last_last_frame, tolerance):
            self.attempt_endtime = time.time()
            self.times.append(self.attempt_endtime - self.attempt_startime)
            
            print(f"Attempt: {self.attempt}, Time: {self.times[-1]}")
            self.attempt += 1
            self.attempt_startime = time.time()
            return True
        return False

    # Isolates a small area that indicates the attempt number 
    # from a frame
    def get_attempt_area(self, frame):
        return frame[0:100, 390:480]        


    def attempt_has_changed(self, frame1_attempt_area, frame_2_attempt_area, tolerance=0):
        diff = frame1_attempt_area - frame_2_attempt_area

        m_norm = np.sum(abs(diff))         # Manhattan norm
        #z_norm = norm(diff.ravel(), 0)    # Zero norm

        if m_norm > tolerance:
            return True
        return False


    def reduce_frame(self, frame):
        return frame[100:, 200:] 



    def key_check(self, key):
        if wapi.GetAsyncKeyState( key):
            return True
        return False

       