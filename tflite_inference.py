# Hight Resolution image shape is 96

# supressing tensorflow warning, info, or things.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from utils import get_lowres_image
import time 
import tqdm, glob 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()

parser.add_argument('--tflite_model_path', type=str, default='pretrained_weights/best_model.tflite')
parser.add_argument('--test_data_path', type=str, default='test/LIME/')
parser.add_argument('--result_data_path', type=str, default='results/tflite/LIME/')
parser.add_argument('--mode', type=str, default="super_resolution")
parser.add_argument('--file_extension', type=str, default='bmp')
parser.add_argument('--visualize_result', type=bool, default=False)


args = parser.parse_args()


def inferrer(image, input_dims=(1, 400, 600, 3)):
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], input_dims)

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, image)

    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    
    # Convert output array to image
    # output_image = (np.squeeze(output, axis=0).clip(0, 1) * 255).astype(np.uint8)
    # img = Image.fromarray(output_image)
    
    return output


def test():

    lowlight_test_images_path = args.test_data_path
    test_files = glob.glob(lowlight_test_images_path + f"*.{args.file_extension}")

    for test_file in tqdm.tqdm(test_files, total=len(test_files)):
        filename = test_file.split("/")[-1]        
        lr_img = Image.open(test_file)

        inputs = (np.asarray(lr_img)/255.0).astype(np.float32)
        inputs = get_lowres_image(inputs, mode=args.mode)
        
        inputs = np.expand_dims(inputs, axis=0)
        t = time.time()
        
        inputs_shape = inputs.shape
        inferrer_input_dims = (1, inputs_shape[0], inputs_shape[1], inputs_shape[2])
        enhanced_image = inferrer(inputs)
       
        print("Time taken for inference: ", time.time() - t)

        if args.visualize_result:
            plt.figure()
            plt.subplot(121)
            plt.title("Low Resolution Image")
            plt.imshow(lr_img)
            
            plt.subplot(122)
            plt.title("Low Resolution Image")
            plt.imshow(enhanced_image[0])
            
            plt.show()
        
        save_file_path = args.result_data_path  + filename
        enhanced_image = tf.cast((enhanced_image[0,:,:,:] * 255), dtype=np.uint8)
        enhanced_image = Image.fromarray(enhanced_image.numpy())
        enhanced_image.save(save_file_path)


if __name__ == '__main__':
    test()
                                                                            