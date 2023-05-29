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
