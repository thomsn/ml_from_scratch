import pickle, gzip, numpy as np
import tensorflow as tf


def convolve(image, filtr, bias, stride):
    rows, columbs = image.shape
    filter_rows, filter_columbs = filtr.shape
    out_rows , out_columbs = int((rows - filter_rows + 1) / stride), \
                             int((columbs - filter_columbs + 1) / stride)
    out = np.zeros((out_rows, out_columbs))
    for out_row in range(out_rows):
        for out_columb in range(out_columbs):
            for row in range(filter_rows):
                for columb in range(filter_columbs):
                    # [performance] may want to cache the stride multiplication ...
                    out[out_row, out_columb] += image[out_row * stride, out_columb * stride] * filtr[row, columb]
                    + bias[row, columb]
    return out

def pool(image, stride):
    rows, columbs = image.shape
    out_rows , out_columbs = int((rows - filter_rows + 1) / stride), \
                             int((columbs - filter_columbs + 1) / stride)
    out = np.zeros((out_rows, out_columbs))
    for out_row in range(out_rows):
        for out_columb in range(out_columbs):
            out[out_row, out_columb] = image[out_row * stride, out_columb * stride]
    return out



print(convolve(np.array([[1,0,0],[0,1,1],[0,1,1]]), np.array([[1,0],[0,1]]), 1))


def load():
    with gzip.open('mnist/mnist.pkl.gz', 'rb') as f:
        return pickle.load(f)

train_set, valid_set, test_set = load()
