# coding:utf-8
"""
@Date : '2020/1/6'
@Desc :  zal
"""
import os

import config
from modules import CNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

cnn = CNN(image_shape=config.image_shape,
          max_captcha=config.max_captcha,
          char_set=config.char_set,
          models_path=config.models_path,
          )

if __name__ == '__main__':
    result = cnn.predict(image=r'D:\python\img_data\text/0B28_W0SkCQX6Mc.png')
    print(result)
