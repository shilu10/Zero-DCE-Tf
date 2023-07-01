import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from tensorflow.keras.layers import Conv2D, Concatenate, SeparableConv2D
from tensorflow.keras import initializers, Model

class ZeroDCENet(keras.Model):
    def __init__(self, n_filters):
        super(ZeroDCENet, self).__init__()
        
        self.conv_1 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_1")
        self.conv_2 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_2")
        self.conv_3 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_3")
        self.conv_4 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_4")
        self.conv_5 = Conv2D(n_filters*2, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_5")
        self.conv_6 = Conv2D(n_filters*2, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_6")
        # concatenation layer
        self.concat = Concatenate(axis=-1, name="concat_layer")
        self.A = Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same", name="curve_params")
        
    def call(self, input_img, training=False):
        conv_1 = self.conv_1(input_img)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_4 = self.conv_4(conv_3)
        
        concat_1 = self.concat([conv_4, conv_3])
        conv_5 = self.conv_5(concat_1)
        
        concat_2 = self.concat([conv_5, conv_2])
        conv_6 = self.conv_6(concat_2)
        
        concat_3 = self.concat([conv_6, conv_1])
        A = self.A(concat_3)
        
        enchanced_image = self.gen_enchanced_image(input_img, A)

        if training:
            return enchanced_image, A 

        return enchanced_image
    
    def gen_enchanced_image(self, input_img, curve_params):
        A = curve_params
        r1 = A[:,:,:,: 3]
        r2 = A[:,:,:,3: 6]
        r3 = A[:,:,:,6: 9]
        r4 = A[:,:,:,9: 12]
        r5 = A[:,:,:,12: 15]
        r6 = A[:,:,:,15: 18]
        r7 = A[:,:,:,18: 21]
        r8 = A[:,:,:,21: 24]
        
        x = input_img + r1 * (tf.pow(input_img,2)-input_img)
        x = x + r2 * (tf.pow(x,2)-x)
        x = x + r3 * (tf.pow(x,2)-x)
        enhanced_image_1 = x + r4*(tf.pow(x,2)-x)
        x = enhanced_image_1 + r5*(tf.pow(enhanced_image_1,2)-enhanced_image_1)		
        x = x + r6*(tf.pow(x,2)-x)	
        x = x + r7*(tf.pow(x,2)-x)
        enhance_image = x + r8*(tf.pow(x,2)-x)
        
        return enhance_image

    def summary(self):
        inputs = Input(shape=(256, 256, 3))
        return Model(inputs=inputs, outputs=self.call(inputs)).summary()



def get_zero_dce(input_shape=(None, None, 3), n_filters=32):
    input_img = Input(shape=input_shape)
    
    conv_1 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_1")(input_img)
    conv_2 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_2")(conv_1)
    
    conv_3 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_3")(conv_2)
    conv_4 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_4")(conv_3)
    concat_1 = Concatenate(axis=-1)([conv_4, conv_3])
 
    conv_5 = Conv2D(n_filters*2, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_5")(concat_1)
    concat_2 = Concatenate(axis=-1)([conv_5, conv_2])
    
    conv_6 = Conv2D(n_filters*2, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_6")(concat_2)
    concat_3 = Concatenate(axis=-1)([conv_6, conv_1])
    
    A = Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same", name="curve_params")(concat_3)
    
    r1 = A[:,:,:,: 3]
    r2 = A[:,:,:,3: 6]
    r3 = A[:,:,:,6: 9]
    r4 = A[:,:,:,9: 12]
    r5 = A[:,:,:,12: 15]
    r6 = A[:,:,:,15: 18]
    r7 = A[:,:,:,18: 21]
    r8 = A[:,:,:,21: 24]
        
    x = input_img + r1 * (tf.pow(input_img,2)-input_img)
    x = x + r2 * (tf.pow(x,2)-x)
    x = x + r3 * (tf.pow(x,2)-x)
    enhanced_image_1 = x + r4*(tf.pow(x,2)-x)
    x = enhanced_image_1 + r5*(tf.pow(enhanced_image_1,2)-enhanced_image_1)		
    x = x + r6*(tf.pow(x,2)-x)	
    x = x + r7*(tf.pow(x,2)-x)
    enhanced_image = x + r8*(tf.pow(x,2)-x)
        
    return Model(inputs=input_img, outputs=enhanced_image)



class ZerodcePlustNet(keras.Model):
    def __init__(self, n_filters, input_img_shape=(256, 256, 3)):
        super(ZerodcePlustNet, self).__init__()
        
        self.input_img_shape = input_img_shape
        self.depth_conv_1 = SeparableConv2D(
            filters=self.n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            depth_multiplier=1,
            depthwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
            pointwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))
            
        self.depth_conv_2 = SeparableConv2D(
            filters=self.n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            depth_multiplier=1,
            depthwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
            pointwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))
            
        self.depth_conv_3 = SeparableConv2D(
            filters=self.n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            depth_multiplier=1,
            depthwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
            pointwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))
            
        self.depth_conv_4 = SeparableConv2D(
            filters=self.n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            depth_multiplier=1,
            depthwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
            pointwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))
            
        self.depth_conv_5 = SeparableConv2D(
            filters=self.n_filters*2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            depth_multiplier=1,
            depthwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
            pointwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))
            
        self.depth_conv_6 = SeparableConv2D(
            filters=self.n_filters*2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            depth_multiplier=1,
            depthwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
            pointwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))
            
        # concatenation layer
        self.concat = Concatenate(axis=-1, name="concat_layer")
        self.A = SeparableConv2D(
            filters=24,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            depth_multiplier=1,
            depthwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
            pointwise_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))
        
    def call(self, input_img, training=False):
        conv_1 = self.depth_conv_1(input_img)
        conv_2 = self.depth_conv_2(conv_1)
        conv_3 = self.depth_conv_3(conv_2)
        conv_4 = self.depth_conv_4(conv_3)
        
        concat_1 = self.concat([conv_4, conv_3])
        conv_5 = self.depth_conv_5(concat_1)
        
        concat_2 = self.concat([conv_5, conv_2])
        conv_6 = self.depth_conv_6(concat_2)
        
        concat_3 = self.concat([conv_6, conv_1])
        A = self.A(concat_3)
        
        enchanced_image = self.gen_enchanced_image(input_img, A)
        
        if training:
            return enchanced_image, A
        return enchanced_image
    
    def gen_enchanced_image(self, input_img, curve_params):
        A = curve_params
        r1 = A[:,:,:,: 3]
        r2 = A[:,:,:,3: 6]
        r3 = A[:,:,:,6: 9]
        r4 = A[:,:,:,9: 12]
        r5 = A[:,:,:,12: 15]
        r6 = A[:,:,:,15: 18]
        r7 = A[:,:,:,18: 21]
        r8 = A[:,:,:,21: 24]
        
        x = input_img + r1 * (tf.pow(input_img,2)-input_img)
        x = x + r2 * (tf.pow(x,2)-x)
        x = x + r3 * (tf.pow(x,2)-x)
        enhanced_image_1 = x + r4*(tf.pow(x,2)-x)
        x = enhanced_image_1 + r5*(tf.pow(enhanced_image_1,2)-enhanced_image_1)		
        x = x + r6*(tf.pow(x,2)-x)	
        x = x + r7*(tf.pow(x,2)-x)
        enhanced_image = x + r8*(tf.pow(x,2)-x)
        
        return enhanced_image

    def summary(self):
        inputs = Input(shape=self.input_img_shape)
        return Model(inputs=inputs, outputs=self.call(inputs)).summary()
