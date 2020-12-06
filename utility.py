import matplotlib.pyplot as plt
import math
import logging
import pickle
from datetime import datetime
from tensorflow import keras

from noisy import *


def plot_image_pairs(image_pairs, image_pair_descs, first_n_pairs = 5):
    assert len(image_pairs) == len(image_pair_descs)
    n_pairs = len(image_pairs)
    fig_size = (4 * n_pairs, 4 * first_n_pairs)
    fig, axes = plt.subplots(first_n_pairs, n_pairs, figsize = fig_size)
    for i, (image_pair, image_pair_desc) in enumerate(zip(image_pairs, image_pair_descs)):
        images = image_pair[:first_n_pairs]
        for idx, image in enumerate(images):
            ax = axes[idx, i]
            ax.imshow(image, cmap = 'gray')
            ax.set_title(image_pair_desc)
    plt.tight_layout()
    plt.show()



def plot_images(images, first_n_pairs = 35):
    n_col = 5
    first_n_pairs = first_n_pairs if first_n_pairs < len(images) else len(images)
    n_row = int(first_n_pairs/n_col)
    images = images[:first_n_pairs]
    figsize = (3.5 * n_col, 4 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize = figsize)
    for idx, image in enumerate(images):
        ax = axes[idx // n_col, idx % n_col]
        ax.imshow(image, cmap = 'gray')
        ax.set_title(f"Weights: {idx}")
    plt.tight_layout()
    plt.show()

    
def get_data(refresh_data = True):
    file_path = "./data/train_test_pair.pkl"
    if refresh_data:
        noise_types = [add_gauss_noise, add_salt_and_pepper_noise, add_poisson_noise, add_speckle_noise]
        train_test_pair = generate_data(noise_types)
        logging.info(f"Saving Data...")
        pickle.dump(train_test_pair, open(file_path, "wb" ) )
    else:
        train_test_pair = pickle.load(open(file_path, "rb" ) )
    return train_test_pair


def generate_data(noise_types):
    def shuffle(a, b):
        assert len(a) == len(b)
        logging.info(f"Shuffling Data...")
        p = np.random.permutation(len(a))
        return a[p], b[p]
    (train, _), (test, _) = keras.datasets.mnist.load_data()
    data = np.append(train, test, axis = 0)
    image_size = data.shape[-2:]
    x, y = np.empty((0, *image_size)), np.empty((0, *image_size))
    for noise_type in noise_types:
        logging.info(f"Adding {noise_type.__name__}...")
        x = np.append(x, np.array([noise_type(a) for a in data]), axis = 0)
        y = np.append(y, data, axis = 0)
    return shuffle(x, y)


def get_train_val_pair(data, train_ratio=0.85, scale=True):
    x, y = data
    assert len(x) == len(y)
    n_images = len(x)
    split_point = int(train_ratio * n_images)
    x_train, y_train, x_val, y_val = x[:split_point,:,:], y[:split_point,:,:], x[split_point:,:,:], y[split_point:,:,:]
    if scale:
        x_train, y_train, x_val, y_val = x_train / 255.0, y_train / 255.0, x_val / 255.0, y_val / 255.0
    return (x_train, y_train), (x_val, y_val)

def measure_nonorthogonality(X):
    # Normalize columns
    X = X / np.linalg.norm(X, axis=0)
    X_inner = X.T @ X
    return (np.abs(X_inner - np.diag(X_inner.diagonal()))).sum() / (X_inner.shape[0]**2 - X_inner.shape[0])