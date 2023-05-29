class Trainer:
    def __init__(self, model, optimzer, ckpt, ckpt_manager, summary_writer, epochs):
        self.model = model 
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager
        self.epochs = epochs
        self.optimzer = optimizer
        
        self.spatial_consistency_loss_tracker = keras.metrics.Mean()
        self.color_constancy_loss_tracker = keras.metrics.Mean()
        self.exposure_loss_tracker = keras.metrics.Mean()
        self.illumination_smoothness_loss_tracker = keras.metrics.Mean()
        self.total_loss_tracker = keras.metrics.Mean()
        
        self.val_spatial_consistency_loss_tracker = keras.metrics.Mean()
        self.val_color_constancy_loss_tracker = keras.metrics.Mean()
        self.val_exposure_loss_tracker = keras.metrics.Mean()
        self.val_illumination_smoothness_loss_tracker = keras.metrics.Mean()
        self.val_total_loss_tracker = keras.metrics.Mean()
        
        self.spatial_consistency_loss_func = SpatialConsistencyLoss()
        self.color_constancy_loss_func = ColorConstancyLoss()
        self.exposure_loss_func = ExposureLoss()
        self.illumination_smoothness_loss_func = IlluminationSmothnessLoss()
        
        log_dir = 'logs/batch_level/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '/train'
        self.train_writer = tf.summary.create_file_writer(log_dir)
    
    def train_step(self, image):
        
        with tf.GradientTape() as tape: 
            enchanced_image, A = self.model(image, training=True)
            loss_res = self.compute_loss(image, enchanced_image, A)
            spatial_constancy_loss, exposure_loss, illumination_loss, color_constancy_loss = loss_res
            total_loss = spatial_constancy_loss + exposure_loss + illumination_loss + color_constancy_loss
            
        params = self.model.trainable_weights
        grads = tape.gradient(total_loss, params)
        
        self.optimizer.apply_gradients(grads, params)
        
        self.spatial_consistency_loss_tracker.update_state(spatialcon_loss)
        self.color_constancy_loss_tracker(color_con_loss)
        self.exposure_loss_tracker(exposure_loss)
        self.illumination_smoothness_loss_tracker(illu_smo_loss)
        self.total_loss_tracker(total_loss)
        
        loss_dict = {
            "total_loss": self.total_loss_tracker.result(),
            "exposure_loss": self.exposure_loss_tracker.result(),
            "color_constancy_loss": self.color_constancy_loss_tracker.result(), 
            "illumination_smoothness_loss": self.illumination_smoothness_loss_tracker.result(),
            "spatial_consistency_loss": self.spatial_consistency_loss_tracker.result()
        }
        
        return loss_dict
    
    def test_step(self, image):
        enchanced_image, A = self.model(image, training=True) 
        loss_res = self.compute_loss(image, enchanced_image, A)
        spatial_constancy_loss, exposure_loss, illumination_loss, color_constancy_loss = loss_res
        total_loss = spatial_constancy_loss + exposure_loss + illumination_loss + color_constancy_loss
        
        self.val_spatial_consistency_loss_tracker.update_state(spatialcon_loss)
        self.val_color_constancy_loss_tracker.update_state(color_con_loss)
        self.val_exposure_loss_tracker.update_state(exposure_loss)
        self.val_illumination_smoothness_loss_tracker.update_state(illu_smo_loss)
        self.val_total_loss_tracker.update_state(total_loss)
        
        loss_dict = {
            "total_loss": self.val_total_loss_tracker.result(),
            "exposure_loss": self.val_exposure_loss_tracker.result(),
            "color_constancy_loss": self.val_color_constancy_loss_tracker.result(), 
            "illumination_smoothness_loss": self.val_illumination_smoothness_loss_tracker.result(),
            "spatial_consistency_loss": self.val_spatial_consistency_loss_tracker.result()
        }
        
        return loss_dict
    
    def compute_loss(self, original, enchanced, curve_params):
        illumination_loss = 200 * self.illumination_smoothness_loss_func(curve_params)
        spatial_constancy_loss = tf.reduce_mean(
            self.spatial_consistency_loss_func(enchanced, original)
        )
        color_constancy_loss = 5 * tf.reduce_mean(self.color_constancy_loss_func(enchanced))
        exposure_loss = 10 * tf.reduce_mean(self.exposure_loss_func(enchanced)) 
        
        return spatial_constancy_loss, exposure_loss, illumination_loss, color_constancy_loss
    
    def load_weights(self, filepath):
        pass 
    
    def save_weights(self, filepath):
        pass 
    
    def train(self, train_ds, val_ds):
        history = defaultdict(list)
        
        for epoch in range(self.epoch): 
            for step, training_batch in tqdm(enumerate(train_ds), total_len=len(train_ds)): 
                loss_dict = self.train_step(training_batch)
                train_loss = loss_dict["total_loss"]
                exposure_loss = loss_dict["exposure_loss"]
                color_constancy_loss = loss_dict["color_constancy_loss"]
                illumination_smoothness_loss = loss_dict["illumination_smoothness_loss"]
                spatial_consistency_loss = loss_dict["spatial_consistency_loss"]

            for step, val_batch in enumerate(val_ds):
                val_loss_dict = self.test_step(val_batch)
                val_loss = loss_dict["total_loss"]
                val_exposure_loss = loss_dict["exposure_loss"]
                val_color_constancy_loss = loss_dict["color_constancy_loss"]
                val_illumination_smoothness_loss = loss_dict["illumination_smoothness_loss"]
                val_spatial_consistency_loss = loss_dict["spatial_consistency_loss"]
            
        history["train_loss"].append(train_loss) 
        history["val_loss"].append(val_loss) 
        history["exposure_loss"].append(exposure_loss) 
        history["val_exposure_loss"].append(val_exposure_loss) 
        history["color_constancy_loss"].append(color_constancy_loss) 
        history["val_color_constancy_loss"].append(val_color_constancy_loss) 
        history["illumination_smoothness_loss"].append(val_illumination_smoothness_loss) 
        history["spatial_consistency_loss"].append(val_loss) 
        history["val_spatial_consistency_loss"].append(train_loss) 
        
        with self.train_writer.as_default(step=epoch):
            tf.summary.scalar('train_loss', train_loss)
            tf.summary.scalar('val_loss', val_loss)
            tf.summary.scalar('exposure_loss', train_loss)
            tf.summary.scalar('val_exposure_loss', val_loss)
            tf.summary.scalar('color_constancy_loss', train_loss)
            tf.summary.scalar('val_color_constancy_loss', val_loss)
            tf.summary.scalar('illumination_smoothness_loss', train_loss)
            tf.summary.scalar('spatial_consistency_loss', val_loss)
            tf.summary.scalar('spatial_consistency_loss', train_loss)
            tf.summary.scalar('val_spatial_consistency_loss', val_loss)
            
        print(f'epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, exposure_loss: {exposure_loss}, val_exposure_loss: {val_exposure_loss}')
    
    def compute_psnr(self):
        pass 
