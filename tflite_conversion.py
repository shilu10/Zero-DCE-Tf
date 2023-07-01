# # Hight Resolution image shape is 96

# supressing tensorflow warning, info, or things.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from tensorflow.keras.models import load_model
import argparse
from models import get_zero_dce, ZerodcePlustNet


parser = argparse.ArgumentParser()

parser.add_argument('--saved_model_path', type=str, default="checkpoint/super_resolution/best_model.h5")
parser.add_argument('--tflite_model_path', type=str, default='checkpoint/super_resolution/best_model.tflite')
parser.add_argument('--num_filters', type=int, default=64)
parser.add_argument('--optimize', type=bool, default=False)
parser.add_argument('--model_type', type=str, default="zerodce")


args = parser.parse_args()

def main():
    if args.model_type == "zerodce":
        model = get_zero_dce(n_filters=args.num_filters)

    if args.model_type == "zerodce_plus":
        model = ZerodcePlustNet(args.num_filters)

    else:
        raise 

    model.load_weights(args.saved_model_path)

    # trained_model = load_model(args.saved_model_path, custom_objects={'tf': tf})
    # weights = trained_model.get_weights()
    # model.set_weights(weights)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(args.tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print("Tflite model is saved at: ", args.tflite_model_path)

if __name__ == '__main__':
    main()         