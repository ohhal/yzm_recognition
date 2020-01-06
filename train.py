# coding:utf-8
"""
@Date : '2020/1/6'
@Desc :  zal
"""
import os

import tensorflow as tf

import config
from modules import CNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(_):
    cnn = CNN(image_shape=config.image_shape,
              max_captcha=config.max_captcha,
              char_set=config.char_set,
              models_path=config.models_path)
    cnn.train(batch_size=config.batch_size,
              train_images_path=config.train_images_path,
              valid_images_path=config.valid_images_path,
              learning_rate=config.learning_rate,
              learning_steps=config.learning_steps)


if __name__ == '__main__':
    tf.app.run()
