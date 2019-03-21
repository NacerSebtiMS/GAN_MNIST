# -*- coding: utf-8 -*-

import gzip
import numpy as np
import matplotlib.pyplot as plt
import gzip

test_image_path = 'data/t10k-images-idx3-ubyte.gz'
test_label_path = 'data/t10k-labels-idx1-ubyte.gz'

train_image_path = 'data/train-images-idx3-ubyte.gz'
train_label_path = 'data/train-labels-idx1-ubyte.gz'


def get_label(n,train=True):
    if train:
        file = gzip.open(train_label_path,'r')
    else :
        file = gzip.open(test_label_path,'r')
        
    file.read(8)
    buff = file.read(1 * n)
    labels = np.frombuffer(buff, dtype=np.uint8).astype(np.int64)
    return labels[-1]


def get_img(n,train=True):
    if train:
        file = gzip.open(train_image_path,'r')
    else :
        file = gzip.open(test_image_path,'r')
        
    image_size = 28
    file.read(16)
    buff = file.read(image_size * image_size * n)
    data = np.frombuffer(buff, dtype=np.uint8).astype(np.float32)
    data = data.reshape(n, image_size, image_size, 1)
    image = np.asarray(data[-1]).squeeze()
    return image

def get_both(n,train=True):
    return get_label(n,train),get_img(n,train)

def test():
    n = 1500
    lbl,img = get_both(n)
    print(lbl)
    plt.imshow(img)

if __name__ == "__main__":
    test()