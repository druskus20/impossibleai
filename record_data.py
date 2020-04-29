from ScreenRecorder import ScreenRecorder 
from keyboard_controller import KeyboardController
import time




def main():

    n = ScreenRecorder((1, 26, 720+1, 576+26))
    
    controller = KeyboardController()
    controller.iniciarJuego()
    time.sleep(10)
    controller.position_window()
    time.sleep(3)

    controller.iniciarPartida()

    time.sleep(2)

    n.capture_live(show=True, save=True, time_frame=60)

    controller.finJuego()
if __name__ == "__main__":
    main()