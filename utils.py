import tensorflow as tf 
import numpy as np 
from tensorflow import keras 
import cv2 

class UnsuuportedFileExtension(Exception):
    def __init__(self, message):
        self.message = message


def random_crop(lr_img):
    lr_crop_size = hr_crop_size
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)


    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]

    return lr_img_cropped


def random_flip(lr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img),
                   lambda: (tf.image.flip_left_right(lr_img)))


def random_rotate(lr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn)


def random_crop_sr(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped  


def get_lowres_image(img, mode="enhancement", scale_factor=4):
    
    if mode == "denoise":
        img = cv2.resize(img, (512, 360))
    
    if mode == "super_resolution":
        img = cv2.resize(img, (400, 260))
        
    if mode == "enhancement":
        img = cv2.resize(img, (600, 400)) 
    
    else:
        img = img 
    
    return img