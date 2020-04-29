import argparse
import os

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from model import get_model
from utils import load_dataset


def train(
        model,
        train_dir: str,
        test_dir: str,
        eval_dir: str,
        output_dir: str,
        batch_size: int,
        num_epoch: int,
        num_load_files_training: int = 10,
        save_checkpoints: bool = True,
        save_best: bool = True,
):
    """
    Train a model
    Input:
    - model: Keras model to train
    - train_dir: Directory where the train files are stored
    - test_dir: Directory where the test files are stored
    - eval_dir: Directory where the eval files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - batch_size: Batch size (Around 10 for 8GB GPU)
    - num_epochs: Number of epochs to do
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set
    Output:
     - float: Accuracy in the development test of the best model
    """

    print("Loading test set")
    X_test, y_test = load_dataset(test_dir)
    print("Loading eval set")
    X_val, y_val = load_dataset(eval_dir)
    print("Loading train set")
    X, y = load_dataset(train_dir)

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'mymodel_{epoch}'),
            save_best_only=save_best,
            monitor='val_loss',
            verbose=1)
    ]

    history = model.fit(X, y, batch_size=batch_size, epochs=num_epoch, validation_data=(X_val, y_val), callbacks=callbacks)

    print('\nhistory dict:', history.history)

    print('\n# Evaluate on test data')
    results = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('test loss, test acc:', results)

    return results[1]


def train_new_model(
        num_epoch,
        train_dir,
        test_dir,
        eval_dir,
        output_dir,
        batch_size=4,
        save_checkpoints=True,
        save_best=True,
):
    print("Loading new model")

    max_acc = train(
        model=get_model((476//2, 520//2), 3, 2),
        train_dir=train_dir,
        test_dir=test_dir,
        eval_dir=eval_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epoch=num_epoch,
        save_checkpoints=save_checkpoints,
        save_best=save_best,
    )

    print(f"Training finished, max accuracy in the development set {max_acc}")


def continue_training(
        num_epoch,
        checkpoint_path,
        train_dir,
        test_dir,
        eval_dir,
        output_dir,
        batch_size=5,
        save_checkpoints=True,
        save_best=True,
):
    model = load_model(checkpoint_path)

    max_acc = train(
        model=model,
        train_dir=train_dir,
        eval_dir=eval_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epoch=num_epoch,
        save_checkpoints=save_checkpoints,
        save_best=save_best,
    )

    print(f"Training finished, max accuracy in the development set {max_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train_new", action="store_true", help="Train a new model",
    )

    group.add_argument(
        "--continue_training",
        action="store_true",
        help="Restore a checkpoint and continue training",
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing the train files",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing the test files",
    )

    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing the eval files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the model and checkpoints are going to be saved",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch size for training (10 for a 8GB GPU seems fine)",
    )

    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of epochs to perform",
    )

    parser.add_argument(
        "--not_save_checkpoints",
        action="store_false",
        help="Do NOT save a checkpoint each epoch (Each checkpoint will rewrite the previous one)",
    )

    parser.add_argument(
        "--not_save_best",
        action="store_false",
        help="Dot NOT save the best model in the development set",
    )

    args = parser.parse_args()

    if args.train_new:
        train_new_model(
            train_dir=args.train_dir,
            eval_dir=args.eval_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epoch=args.num_epochs,
            save_checkpoints=args.not_save_checkpoints,
            save_best=args.not_save_best,
        )
