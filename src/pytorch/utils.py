import datetime
import glob
import logging
import os
from typing import Iterable, Sized, List, Set

import numpy as np

try:
    import cupy as cp

    cupy = True
except ModuleNotFoundError:
    cupy = False
    logging.warning(
        "Cupy not found, dataset preprocessing is going to be slow. "
        "Installing copy is highly recommended (x10 speedup): "
        "https://docs-cupy.chainer.org/en/latest/install.html?highlight=cuda90#install-cupy"
    )


def check_valid_y(data: np.ndarray) -> bool:
    """
    Check if any key has been pressed in the datased. Some files may not have any key recorded due to windows
    permission errors on some computers, people not using WASD or other problems, we want to discard these files.
    Input:
     - data: ndarray [num_examples x 6]
    Output:
    - Bool: True if the file is valid, False is there no key recorded
    """
    seen_keys: Set[int] = set()
    for i in range(0, data.shape[0]):
        if np.array_equal(data[i][5], [0]):
            seen_keys.add(0)
        elif np.array_equal(data[i][5], [1]):
            seen_keys.add(1)

        if len(seen_keys) >= 2:
            return True

    else:
        return False


def reshape_y(data: np.ndarray) -> np.ndarray:
    """
    Get gold values from data. multi-hot vector to one-hot vector
    Input:
     - data: ndarray [num_examples x 6]
    Output:
    - ndarray [num_examples]
    """
    reshaped = np.zeros(data.shape[0], dtype=np.int16)
    for i in range(0, data.shape[0]):
        if np.array_equal(data[i][5], [0]):
            reshaped[i] = 0
        elif np.array_equal(data[i][5], [1]):
            reshaped[i] = 1
    return reshaped


def reshape_x_numpy(
        data: np.ndarray, dtype=np.float16) -> np.ndarray:
    """
    Get images from data as a list and preprocess them.
    Input:
     - data: ndarray [num_examples x 6]
     -dtype: numpy dtype for the output array
      from the sequence of images
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]
    """
    mean = np.array([0.485, 0.456, 0.406], dtype)
    std = np.array([0.229, 0.224, 0.225], dtype)
    reshaped = np.zeros((len(data) * 5, 3, 270, 480), dtype=dtype)
    for i in range(0, len(data)):
        for j in range(0, 5):
            img = np.array(data[i][j], dtype=dtype) / 255
            reshaped[i * 5 + j] = np.rollaxis((img - mean) / std, 2)

    return reshaped


if cupy:
    def reshape_x_cupy(data: np.ndarray, dtype=cp.float16) -> np.ndarray:
        """
        Get images from data as a list and preprocess them (using GPU).
        Input:
         - data: ndarray [num_examples x 6]
         -dtype: numpy dtype for the output array
          from the sequence of images
        Output:
        - ndarray [num_examples * 5, num_channels, H, W]
        """

        mean = cp.array([0.485, 0.456, 0.406], dtype=dtype)
        std = cp.array([0.229, 0.224, 0.225], dtype=dtype)
        reshaped = np.zeros((len(data) * 5, 3, 270, 480), dtype=dtype)
        for i in range(0, len(data)):
            for j in range(0, 5):
                img = cp.array(data[i][j], dtype=dtype) / 255
                reshaped[i * 5 + j] = cp.asnumpy(cp.rollaxis((img - mean) / std, 2))
        return reshaped


def reshape_x(data: np.ndarray, fp=16) -> np.ndarray:
    """
    Get images from data as a list and preprocess them, if cupy is available it uses the GPU,
    else it uses the CPU (numpy)
    Input:
     - data: ndarray [num_examples x 6]
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]
    """
    if cupy:
        if fp == 16:
            return reshape_x_cupy(data, dtype=cp.float16)
        elif fp == 32:
            return reshape_x_cupy(data, dtype=cp.float32)
        elif fp == 64:
            return reshape_x_cupy(data, dtype=cp.float64)
        else:
            raise ValueError(
                f"Invalid floating-point precision: {fp}: Available values: 16, 32, 64"
            )
    else:
        if fp == 16:
            return reshape_x_numpy(data, dtype=np.float16)
        elif fp == 32:
            return reshape_x_numpy(data, dtype=np.float32)
        elif fp == 64:
            return reshape_x_numpy(data, dtype=np.float64)
        else:
            raise ValueError(
                f"Invalid floating-point precision: {fp}: Available values: 16, 32, 64"
            )


def batch(iterable: Sized, n: int = 1) -> Iterable:
    """
    Given a iterable generate batches of size n
    Input:
     - Sized that will be batched
     - n: Integer batch size
    Output:
    - Iterable
    """
    l: int = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


def nn_batchs(X: Sized, y: Sized, n: int = 1, sequence_size: int = 5) -> Iterable:
    """
    Given the input examples and the golds generate batches of sequence_size
    Input:
     - X: Sized input examples
     - y: Sized golds
     - n: Integer batch size
     -sequence_size: Number of images in a training example. len(x) = len(y) * sequence_size
    Output:
    - Iterable
    """

    assert len(X) == len(y) * sequence_size, (
        f"Inconsistent data, len(X) must equal len(y)*sequence_size."
        f" len(X)={len(X)}, len(y)={len(y)}, sequence_size={sequence_size}"
    )
    bg_X: Iterable = batch(X, n * sequence_size)
    bg_y: Iterable = batch(y, n)

    for b_X, bg_y in zip(bg_X, bg_y):
        yield b_X, bg_y


def load_file(
        path: str, fp: int = 16
) -> (np.ndarray, np.ndarray):
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
        logging.warning(
            f"[Unknown exception, probably corrupted file] Error in file: {path}, ignoring the file."
        )
        return np.array([]), np.array([])

    if check_valid_y(data):
        X = reshape_x(data, fp)
        y = reshape_y(data)
        return X, y

    else:
        logging.warning(f"Invalid file, no keys recorded: {path}, ignoring the file.")
        return np.array([]), np.array([])


def load_dataset(path: str, fp: int = 32) -> (np.ndarray, np.ndarray):
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
        X_batch, y_batch = load_file(file, fp)
        if len(X_batch) > 0 and len(y_batch) > 0:
            if len(X) == 0:
                X = X_batch
                y = y_batch
            else:
                X = np.concatenate((X, X_batch), axis=0)
                y = np.concatenate((y, y_batch), axis=0)

    if len(X) == 0 or len(y) == 0:
        # Since this function is used for loading the dev and test set, we want to stop the execution if we don't
        # have a valid test of dev set.
        raise ValueError(f"Empty dataset, all files invalid. Path: {path}")

    return X, y


def load_and_shuffle_datasets(paths: List[str], fp: int = 32) -> (np.ndarray, np.ndarray):
    """
    Load multiple dataset files and shuffle the data, useful for training
    Input:
     - paths: List of paths to dataset files
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - X: input examples [num_examples_per_file * num_files, 5, 3, H, W]
    - y: golds for the input examples [num_examples_per_file * num_files]
    """
    data_array: np.ndarray = np.array([])

    for file_no, file in enumerate(paths):
        # print(f"Loading file {file_no+1} of {len(paths)}...")
        try:
            data: np.ndarray = np.load(file, allow_pickle=True, fp=fp)["arr_0"]
        except (IOError, ValueError) as err:
            logging.warning(f"[{err}] Error in file: {file}, ignoring the file.")
            continue
        except:
            logging.warning(
                f"[Unknown exception, probably corrupted file] Error in file: {file}, ignoring the file."
            )
            continue

        if check_valid_y(data):
            if len(data_array) == 0:
                data_array = data
            else:
                data_array = np.concatenate((data_array, data), axis=0)
        else:
            logging.warning(
                f"Invalid file, no keys recorded: {file}, ignoring the file."
            )

    if len(data_array) > 0:
        np.random.shuffle(data_array)
    else:
        # Since this function is used for training, we want to continue training with the next files,
        # so we return two empty arrays
        logging.warning(f"Empty dataset, all files invalid. Path: {paths}")
        return np.array([]), np.array([])

    X: np.ndarray = reshape_x(data_array, fp)
    y: np.ndarray = reshape_y(data_array)

    return X, y


def print_trace(message: str) -> None:
    """
    Print a message in the <date> : message format
    Input:
     - message: string to print
    Output:
    """
    print("<" + str(datetime.datetime.now()) + ">  " + str(message))
