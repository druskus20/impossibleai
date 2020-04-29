from keyboard_controller import KeyboardController
import time

controller = KeyboardController()

controller.iniciarJuego()
time.sleep(10)
controller.position_window()
time.sleep(3)
print("COMENZAMOS")
controller.iniciarPartida()
time.sleep(2)
print("SALTAMOS")
controller.accion(1)
time.sleep(5)
print("REPETIMOS")
controller.accion(1)
time.sleep(1)
controller.accion(0)
print("FIN")

controller.finJuego()