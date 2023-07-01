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
parser.add_argument('--checkpoint_filepath', type=str, default="checkpoint/enhancement/")
parser.add_argument('--n_filters', type=int, default=32)
parser.add_argument('--summary', type=bool, default=False)
parser.add_argument('--store_model_summary', type=bool, default=False)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataloader = DataLoader("lol")
    dataloader.initialize()
    train_ds = dataloader.get_dataset(
                    subset="train",
                    batch_size=args.batch_size,
                    transform=True
                )

    val_ds = dataloader.get_dataset(
                    subset="val",
                    batch_size=args.batch_size,
                    transform=False
                )
    
    model = get_enhancement_model(
            num_rrg=args.num_rrg,
            num_mrb=args.num_mrb,
            num_channels=args.num_channels
        )

    if args.summary:
        model.summary()

    if args.store_model_summary:
        tf.keras.utils.plot_model(to_file="mirnet_enhancement.png")

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="val_psnr_enhancement",
            patience=10,
            mode='max'
        )

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            args.checkpoint_filepath+"best_model.h5",
            monitor="val_psnr_enhancement",
            save_weights_only=True,
            mode="max",
            save_best_only=True,
            period=1
        )

    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(
            monitor='val_psnr_enhancement',
            factor=0.5,
            patience=5,
            verbose=1,
            epsilon=1e-7,
            mode='max'
        )

    if args.loss_function == "charbonnier":
        loss_func = charbonnier_loss

    if args.loss_function == "l1":
        loss_func = tf.keras.losses.MeanAbsoluteError()

    else:
        loss_func = tf.keras.losses.MeanSquaredError()

    if args.use_custom_trainer:
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
            epoch=tf.Variable(1)
        )

        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=args.checkpoint_filepath + "custom_training/",
            max_to_keep=5
        )

        status = checkpoint.restore(manager.latest_checkpoint)
        trainer = Trainer(
                    model=model,
                    loss_func=loss_func,
                    metric_func=psnr_enhancement,
                    optimizer=optimizer,
                    ckpt=checkpoint,
                    ckpt_manager=manager,
                    epochs=args.n_epochs,
                    mode="enhancement"
                )

        trainer.train(train_ds, val_ds)

    else:
        model.compile(
                optimizer=optimizer,
                loss=loss_func,
                metrics=[psnr_enhancement]
            )

        model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.n_epochs,
                callbacks=[early_stopping_callback, model_checkpoint_callback, reduce_lr_loss]
            )


if __name__ == '__main__':
    train()
