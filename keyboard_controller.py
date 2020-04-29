import os
import time
from subprocess import Popen
import win32gui
import win32api
import win32con
import ctypes
import time
import pynput
from pynput.keyboard import Key, Controller


SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ ) 
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


class KeyboardController:
    proc = None

    def iniciarJuego(self):
        win32api.SetCursorPos((1920,1080))
        #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, 0, 0)
        self.proc = Popen('ImpossibleGame.exe')
        
    def position_window(self):

        appname = 'Impossible Game'
        xpos = 0
        ypos = 0
        width = 720
        length = 576
        

        def enumHandler(hwnd, lParam):
            if win32gui.IsWindowVisible(hwnd):
                if appname in win32gui.GetWindowText(hwnd):
                    win32gui.MoveWindow(hwnd, xpos, ypos, width, length, True)


        win32gui.EnumWindows(enumHandler, None)
    
    def accion(self, x):
        if (x == 0 and self.x != 0):
            self.x = 0
            ReleaseKey(0x39)
        if (x == 1 and self.x != 1):
            self.x = 1
            PressKey(0x39)
    
    def iniciarPartida(self):
        PressKey(0x1C)
        time.sleep(1)
        ReleaseKey(0x1C)
    
    def finJuego(self):
        self.proc.kill()

    def __init__(self):
        self.x = 0
        self.keyboard = Controller()

