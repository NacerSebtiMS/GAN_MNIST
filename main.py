# -*- coding: utf-8 -*-

import gzip
import numpy as np
import matplotlib.pyplot as plt
import gzip

test_image_path = 'data/t10k-images-idx3-ubyte.gz'
test_label_path = 'data/t10k-labels-idx1-ubyte.gz'

train_image_path = 'data/train-images-idx3-ubyte.gz'
train_label_path = 'data/train-labels-idx1-ubyte.gz'


def label(file,n):
	f.read(8)
	buff = f.read(1 * n)
	labels = np.frombuffer(buff, dtype=np.uint8).astype(np.int64)
	return labels[-1]


def img(file,n):
	image_size = 28
	f.read(16)
	buff = f.read(image_size * image_size * n)
	data = np.frombuffer(buff, dtype=np.uint8).astype(np.float32)
	data = data.reshape(n, image_size, image_size, 1)
	image = np.asarray(data[-1]).squeeze()
	return image