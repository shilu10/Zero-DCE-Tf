from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
from imutils import paths 
import os 
import shutil
from utils import UnsuuportedFileExtension

class DataLoader:
    def __init__(self, dname: str, resize_shape: int, batch_size: int):
        assert dname in ["lol"], "given dataset name is not valid, supported datasets are ['lol']"  
        assert type(resize_shape) == int, 'Unknown dtype for resize shape, needed Int' 
        assert type(batch_size) == int, 'Unknown dtype for batch_size, needed Int' 
        self.dname = dname 
        self.resize_shape = resize_shape
        self.batch_size = batch_size
    
    def initialize(self):
        try: 
            if self.dname == "lol":
                LOL_DATA_URL = 'https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6'
                if not os.path.exists('lol_dataset'):
                    gdown.download(LOL_DATA_URL, quiet=True)
                    os.system(f'unzip -q lol_dataset')

                if (os.path.exists("lol_dataset.zip") and not os.path.exists("lol_dataset")):
                    os.system(f'unzip -q lol_dataset')
                    
        except Exception as err:
            return err 
        
        else:
            self.train_data_path = os.path.join("lol_dataset", "our485", "low")
            self.val_data_path = os.path.join("lol_dataset", "eval15", 'low')
        
    def __preprocess_image(self, img_fpath): 
        try: 
            f_ext = None
            raw = tf.io.read_file(img_fpath)
            if f_ext == 'jpeg':
                image = tf.image.decode_jpeg(raw)
            if f_ext == 'png':
                image = tf.image.decode_png(raw)
            if f_ext == "jpg":
                image = tf.image.decode_jpeg(raw)
            else:
                image = tf.image.decode_png(raw)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, size=[self.resize_shape, self.resize_shape],
                                                        method=tf.image.ResizeMethod.BICUBIC)
            image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
            return image
            
        except Exception as err:
            return err
        
    def get_dataset(self, split_name):
        assert split_name in ("train", 'val'), "unsupported split type"
        try:
            if split_name == "train":
                train_files = list(paths.list_files(self.train_data_path))
                tf_ds = tf.data.Dataset.from_tensor_slices(train_files)
                tf_ds = tf_ds.map(self.__preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache()
                tf_ds = tf_ds.shuffle(buffer_size=50)
                tf_ds = tf_ds.batch(self.batch_size)
                tf_ds = tf_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                return tf_ds
            
            else:
                val_files = list(paths.list_files(self.val_data_path))
                tf_ds = tf.data.Dataset.from_tensor_slices(val_files)
                tf_ds = tf_ds.map(self.__preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache()
                tf_ds = tf_ds.shuffle(buffer_size=5)
                tf_ds = tf_ds.batch(self.batch_size)
                tf_ds = tf_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                return tf_ds
                
        except Exception as err:
            return err
