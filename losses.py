import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import AveragePooling2D, Conv2D
import numpy as np 
import os 
import sys 


class ColorConstancyLoss(keras.losses.Loss):
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()
    
    def __call__(self, image):
        super.__call__()
        # mean value calculation of the image.
        rgb_mean = tf.reduce_mean(image, axis=(1, 2), keepdims=True)
        red_mean = rgb_mean[:, :, :, 0]
        green_mean = rgb_mean[:, :, :, 1]
        blue_mean = rgb_mean[:, :, :, 2]
        
        diff_rb = tf.square(red_mean - blue_mean)
        diff_rg = tf.square(red_mean - green_mean)
        diff_gb = tf.square(green_mean - blue_mean)
        
        #squared differnece 
        diff_rb = tf.square(diff_rb)
        diff_rg = tf.square(diff_rg)
        diff_gb = tf.square(diff_gb)
        
        loss_val = tf.sqrt(diff_rb + diff_rg + diff_gb)
        return loss_val


class ExposureLoss(keras.losses.Loss):
    def __init__(self, mean_val):
        super(ExposureLoss, self).__init__()
        self.mean_val = mean_val
    
    def __call__(self, image):
        super.__call__()
        x = tf.reduce_mean(image, 3, keepdims=True)
        mean = AveragePooling2D(pool_size=16, strides=16)(x) # non overlap
        
        d = tf.reduce_mean(tf.pow(mean- self.mean_val, 2))
        
        return d


class IlluminationSmothnessLoss(keras.losses.Loss):
    def __init__(self, tvloss_weights=1):
        super(IlluminationSmothnessLoss, self).__init__()
        self.tvloss_weights = tvloss_weights
        
    def __call__(self, x):
        super.__call__()
        batch_size = tf.shape(x)[0]
        h_x = tf.shape(x)[1]
        w_x = tf.shape(x)[2]
        count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
        count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
        h_tv = tf.reduce_sum(tf.square((x[:, 1:, :, :] - x[:, : h_x - 1, :, :])))
        w_tv = tf.reduce_sum(tf.square((x[:, :, 1:, :] - x[:, :, : w_x - 1, :])))
        batch_size = tf.cast(batch_size, dtype=tf.float32)
        count_h = tf.cast(count_h, dtype=tf.float32)
        count_w = tf.cast(count_w, dtype=tf.float32)
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size




class SpatialConsistencyLoss(keras.losses.Loss):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
    
    def init_f_left(self, shape, dtype=None):
        ker = np.zeros(shape, dtype=None)
        ker[1][0] = -1
        ker[1][1] = 1
        return ker

    def init_f_right(self, shape, dtype=None):
        ker = np.zeros(shape, dtype=None)
        ker[1][2] = -1
        ker[1][1] = 1
        return ker

    def init_f_up(self, shape, dtype=None):
        ker = np.zeros(shape, dtype=None)
        ker[0][1] = -1
        ker[1][1] = 1
        return ker

    def init_f_down(self, shape, dtype=None):
        ker = np.zeros(shape, dtype=None)
        ker[2][1] = -1
        ker[1][1] = 1
        return ker
    
    
    def call(self, orginal, enchanced): 
        orginal_mean = tf.reduce_mean(orginal, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(enchanced, 3, keepdims=True)

        original_pool =  AveragePooling2D(pool_size=4)(orginal_mean)		
        enhanced_pool = AveragePooling2D(pool_size=4)(enhanced_mean)
        
        # original
        d_original_left = Conv2D(kernel_initializer=self.init_f_left, 
                                padding='same', kernel_size=(3,3), filters=1)(original_pool)
        d_original_right = Conv2D(kernel_initializer=self.init_f_right,
                                padding='same', kernel_size=(3,3), filters=1)(original_pool)
        d_original_up = Conv2D(kernel_initializer=self.init_f_up,
                                padding='same', kernel_size=(3,3), filters=1)(original_pool)
        d_original_down = Conv2D(kernel_initializer=self.init_f_down,
                                padding='same', kernel_size=(3,3), filters=1)(original_pool)
        
        # enchanced
        d_enhanced_letf = Conv2D(kernel_initializer=self.init_f_left,
                                padding='same', kernel_size=(3,3), filters=1)(enhanced_pool)
        d_enhanced_right = Conv2D(kernel_initializer=self.init_f_right,
                                padding='same', kernel_size=(3,3), filters=1)(enhanced_pool)
        d_enhanced_up = Conv2D(kernel_initializer=self.init_f_up,
                                padding='same', kernel_size=(3,3), filters=1)(enhanced_pool)
        d_enhanced_down = Conv2D(kernel_initializer=self.init_f_down,
                                padding='same', kernel_size=(3,3), filters=1)(enhanced_pool)

        d_left = tf.pow(d_original_left - d_enhanced_letf, 2)
        d_right = tf.pow(d_original_right - d_enhanced_right, 2)
        d_up = tf.pow(d_original_up - d_enhanced_up, 2)
        d_down = tf.pow(d_original_down - d_enhanced_down, 2)
        E = (d_left + d_right + d_up + d_down)

        return E
