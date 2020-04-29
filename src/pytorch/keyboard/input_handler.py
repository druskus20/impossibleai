# inputsHandler.py

# Authors: Iker García Ferrero and Eritz Yerga

from .keyboard_controller import SPACE
from .keyboard_controller import press_key, release_key


def release_all_keys() -> None:
    """
     Release all keys
     """
    release_key(SPACE)


def pres_space() -> None:
    """
       Release all keys and push Space
   """
    press_key(SPACE)


def select_key(key) -> None:
    """
      Given a ket in integer format, send to windows the virtual ket push
      """
    if key == 0:
        release_all_keys()
    elif key == 1:
        pres_space()
