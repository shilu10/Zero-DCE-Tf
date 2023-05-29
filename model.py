class ZeroDCENet(keras.Model):
    def __init__(self, n_filters):
        super(ZeroDCENet, self).__init__()
        
        self.conv_1 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_1")
        self.conv_2 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_2")
        self.conv_3 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_3")
        self.conv_4 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_4")
        self.conv_5 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_5")
        self.conv_6 = Conv2D(n_filters, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv_6")
        # concatenation layer
        self.concat = Concatenate(axis=-1, name="concat_layer")
        self.A = Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same", name="curve_params")
        
    def call(self, inputs):
        conv_1 = self.conv_1(inputs)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_4 = self.conv_4(conv_3)
        
        concat_1 = self.concat([conv_4, conv_3])
        conv_5 = self.conv_5(concat_1)
        
        concat_2 = self.concat([conv_5, conv_2])
        conv_6 = self.conv_6(concat_2)
        
        concat_3 = self.concat([conv_6, conv_1])
        A = self.A(concat_3)
        
        return A
    
    def gen_enchanced_mage(self, curve_params):
        A = curve_params
        
    
    def summary(self):
        inputs = Input(shape=(256, 256, 3))
        return Model(inputs=inputs, outputs=self.call(inputs)).summary()
