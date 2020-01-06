# -*- coding: utf-8 -*-
"""
@Date : '2020/1/6'
@Desc :  zal
"""
char_set = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
    'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]  # 待训练标签字符
image_shape = [30, 150, 3]  # 图像尺寸
max_captcha = 4  # 文本的长度

train_images_path = r'D:\github\yzm_recognition\img_data\train'  # 训练集图像存放路径
valid_images_path = r'D:\github\yzm_recognition\img_data\valid'  # 验证集图像存放路径
models_path = r'D:\github\yzm_recognition\model/'  # 模型存放路径

learning_rate = 1e-3  # 学习率
batch_size = 32  # batch_size
learning_steps = 1500  # 学习步数

predict_batch_size = 32
predict_images_path = r'D:\python\img_data\text'  # 待检测图像存放路径
