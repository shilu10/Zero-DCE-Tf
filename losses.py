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
