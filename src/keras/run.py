import argparse
import threading
import time
from tkinter import *

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import reshape_x

from keyboard.input_handler import select_key
from keyboard.keys import key_check, key_press
from screen import recorder as screen_recorder


def run_keras(model_dir, show_current_control):
    """
    Generate dataset exampled from a human playing a videogame
    HOWTO:
        Set your game in windowed mode
        Set your game to 1600x900 resolution
        Move the game window to the top left corner, there should be a blue line of 1 pixel in the left bezel of your
         screen and the window top bar should start in the top bezel of your screen.
        Let the AI play the game!
    Controls:
        Push QE to exit
        Push L to see the input images
        Push and hold J to use to use manual control
    Input:
    - model_dir: Directory where the model to use is stored (model.bin and model_hyperparameters.json files)
    - show_current_control: Show a window with text that indicates if the car is currently being driven by
      the AI or a human
    Output:
    """

    show_what_ai_sees: bool = False
    fp16: bool
    model = load_model(model_dir)
    stop_recording: threading.Event = threading.Event()

    th_img: threading.Thread = threading.Thread(target=screen_recorder.img_thread, args=[stop_recording])
    th_img.start()
    th_seq: threading.Thread = threading.Thread(target=screen_recorder.image_sequencer_thread, args=[stop_recording])
    # Wait to launch the image_sequencer_thread, it needs the img_thread to be running
    time.sleep(5)
    th_seq.start()

    if show_current_control:
        root = Tk()
        var = StringVar()
        var.set("T.E.D.D. 1104 Driving")
        l = Label(root, textvariable=var, fg="green", font=("Courier", 44))
        l.pack()

    last_time = time.time()
    model_prediction = np.asarray([0])
    while True:
        img_seq = screen_recorder.seq.copy()
        keys = key_check()
        if not "J" in keys:
            X = reshape_x(np.array([img_seq]))
            model_prediction = model.predict(X).argmax(axis=-1)
            select_key(model_prediction)

            if show_current_control:
                var.set("T.E.D.D. 1104 Driving")
                l.config(fg="green")
                root.update()
        else:
            if show_current_control:
                var.set("Manual Control")
                l.config(fg="red")
                root.update()

        if show_what_ai_sees:
            cv2.imshow("window1", img_seq[0])
            cv2.imshow("window2", img_seq[1])
            cv2.imshow("window3", img_seq[2])
            cv2.imshow("window4", img_seq[3])
            cv2.imshow("window5", img_seq[4])

        if "Q" in keys and "E" in keys:
            print("\nStopping...")
            stop_recording.set()
            th_seq.join()
            th_img.join()
            if show_what_ai_sees:
                cv2.destroyAllWindows()
            break

        if "L" in keys:
            time.sleep(0.1)  # Wait for key release
            if show_what_ai_sees:
                cv2.destroyAllWindows()
                show_what_ai_sees = False
            else:
                show_what_ai_sees = True

        time_it = time.time() - last_time
        print(
            f"Recording at {screen_recorder.fps} FPS\n"
            f"Actions per second {None if time_it == 0 else 1 / time_it}\n"
            f"Key predicted by nn: {key_press(model_prediction)}\n"
            f"Push QE to exit\n"
            f"Push L to see the input images\n"
            f"Push J to use to use manual control\n",
            end="\r",
        )
        last_time = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the model to use is stored",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 floating point precision: "
             "Requires Nvidia Apex: https://www.github.com/nvidia/apex "
             "and a modern Nvidia GPU FP16 capable (Volta, Turing and future architectures).",
    )

    parser.add_argument(
        "--show_current_control",
        action="store_true",
        help="Show a window with text that indicates if the car is currently being driven by the AI or a human",
    )

    parser.add_argument(
        "--evasion_score",
        type=float,
        default=200,
        help="Mean squared error value between images to activate the evasion maneuvers",
    )

    args = parser.parse_args()

    screen_recorder.initialize_global_variables()

    run_keras(
        model_dir=args.model_dir,
        show_current_control=args.show_current_control,
    )
