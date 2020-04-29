import glob
import os

import numpy as np
from tensorflow.keras.utils import to_categorical

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def reshape_x(data):
    """
    Get images from data as a list and preprocess them.
    Input:
     - data: ndarray [num_examples x 6]
     -dtype: numpy dtype for the output array
      from the sequence of images
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]
    """

    
    ims_seqs = []
    for i in range(0, len(data)):
        frames = []
        for j in range(0, 3):

            img = np.array(data[i][j], dtype=np.float32) / 255
            frames.append((img - mean) / std)
        ims_seqs.append(frames)
        del frames
    ims_seqs = np.asarray(ims_seqs)
    return ims_seqs


def load_file(path):
    """
    Load dataset from file: Load, reshape and preprocess data.
    Input:
     - path: Path of the dataset
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - X: input examples [num_examples, 5, 3, H, W]
    - y: golds for the input examples [num_examples]
    """
    try:
        data = np.load(path, allow_pickle=True)["arr_0"]
    except (IOError, ValueError) as err:
        logging.warning(f"[{err}] Error in file: {path}, ignoring the file.")
        return np.array([]), np.array([])
    except:
        logging.warning(f"[Unknown exception, probably corrupted file] Error in file: {path}, ignoring the file.")
        return np.array([]), np.array([])

    X = reshape_x(data)
    y = to_categorical(data[:, 3])
    del data
    return X, y


def load_dataset(path: str) -> (np.ndarray, np.ndarray):
    """
    Load dataset from directory: Load, reshape and preprocess data for all the files in a directory.
    Input:
     - path: Path of the directory
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - X: input examples [num_examples_per_file * num_files, 5, 3, H, W]
    - y: golds for the input examples [num_examples_per_file * num_files]
    """
    X: np.ndarray = np.array([])
    y: np.ndarray = np.array([])

    files = glob.glob(os.path.join(path, "*.npz"))
    for file_n, file in enumerate(files):
        print(f"Loading file {file_n + 1} of {len(files)}...")
        X_batch, y_batch = load_file(file)
        if len(X_batch) > 0 and len(y_batch) > 0:
            if len(X) == 0:
                X = X_batch
                y = y_batch
            else:
                X = np.concatenate((X, X_batch), axis=0)
                y = np.concatenate((y, y_batch), axis=0)
        del X_batch, y_batch

    if len(X) == 0 or len(y) == 0:
        # Since this function is used for loading the dev and test set, we want to stop the execution if we don't
        # have a valid test of dev set.
        raise ValueError(f"Empty dataset, all files invalid. Path: {path}")

    return X, y
