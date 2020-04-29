import win32api as wapi

all_keys = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    all_keys.append(char)


def key_check():
    keys = []
    for key in all_keys:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def key_press(key):
    if key == 1:
        return " "
    return "none"
