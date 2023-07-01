import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil, glob 
import sys 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from models import ZeroDCENet, ZeroDCELiteNet
import argparse
from utils import UnsuuportedFileExtension
from dataloader import DataLoader
from custom_trainer import Trainer
from losses import IlluminationSmothnessLoss, ExposureLoss, SpatialConsistencyLoss, ColorConstancyLoss


parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint_filepath', type=str, default="checkpoint/")
parser.add_argument('--n_filters', type=int, default=32)
parser.add_argument('--summary', type=bool, default=False)
parser.add_argument('--store_model_summary', type=bool, default=False)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--resize_dim', type=int, default=256)


args = parser.parse_args()

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataloader = LOLDataLoader("lol", args.resize_dim)
    dataloader.initialize()
    train_ds = dataloader.get_dataset(
                    subset="train",
                    batch_size=args.batch_size,
                    transform=False
                )

    val_ds = dataloader.get_dataset(
                    subset="val",
                    batch_size=args.batch_size,  # by default batch_size=1 for val_ds, even if we specify batch_size=32.
                    transform=False
                )
    
    model = ZeroDCENet(args.n_filters)

    if args.summary:
        model.summary()

    if args.store_model_summary:
        tf.keras.utils.plot_model(to_file="zerodcenet_enhancement.png")

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    spatial_consistency_loss = SpatialConsistencyLoss()
    exposure_loss = ExposureLoss(mean_val=0.6)
    illumination_smoothness_loss = IlluminationSmothnessLoss()
    color_constancy_loss = ColorConstancyLoss()

    
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model,
        epoch=tf.Variable(1)
    )

    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=args.checkpoint_filepath,
        max_to_keep=5
    )

    status = checkpoint.restore(manager.latest_checkpoint)
    trainer = Trainer(model=model, 
                optimizer=optimizer, 
                spatial_consistency_loss=spatial_consistency_loss, 
                color_constancy_loss=color_constancy_loss,
                exposure_loss=exposure_loss, 
                illumination_smoothness_loss=illumination_smoothness_loss, 
                ckpt=checkpoint, 
                ckpt_manager=manager, 
                epochs=100
            )

    trainer.train(train_ds, val_ds)

if __name__ == '__main__':
    train()
