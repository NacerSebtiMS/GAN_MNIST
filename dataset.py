# -*- coding: utf-8 -*-

import gzip
import numpy as np
import matplotlib.pyplot as plt
import gzip

test_image_path = 'data/t10k-images-idx3-ubyte.gz'
test_label_path = 'data/t10k-labels-idx1-ubyte.gz'

train_image_path = 'data/train-images-idx3-ubyte.gz'
train_label_path = 'data/train-labels-idx1-ubyte.gz'

#----------------------------------------------------------------------------
# Direct interaction with the dataset


# Getting labels
def get_label(n,train=True):
    if train:
        file = gzip.open(train_label_path,'r')
    else :
        file = gzip.open(test_label_path,'r')
        
    file.read(8)
    buff = file.read(1 * n)
    labels = np.frombuffer(buff, dtype=np.uint8).astype(np.int64)
    return labels[-1]

# Getting images
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

# Getting both the image and the label at given index from the dataset
def get_both(n,train=True):
    return get_label(n,train),get_img(n,train)

# Returns n iamges with the given label
def get_img_by_lbl(n,label,train=True):
    cpt=1
    imgs = []
    while n>len(imgs):
        if get_label(cpt,train) == label:
            imgs += [get_img(cpt,train)]
        cpt+=1
    return imgs

#----------------------------------------------------------------------------
# Functions used to print the results

def show_img(img):
    plt.imshow(img)

def show_mult_img(imgs):
    n = len(imgs)
    l = int(n**(1/2))
    if int(n**(1/2)) != n**(1/2):
        l+=1
    fig = plt.figure(figsize=(l*2,l*2))
    for i in range(len(imgs)):
        sub = fig.add_subplot(l,l,i+1)
        sub.imshow(imgs[i], interpolation='nearest')

#----------------------------------------------------------------------------
# Testing Area
        
def test():
#    n = 15
#    lbl,img = get_both(n)
#    print(lbl)
#    plt.imshow(img)
    n = 16
    label = 4
    imgs = get_img_by_lbl(n,label)
    i=0
    show_mult_img(imgs)

if __name__ == "__main__":
    test()