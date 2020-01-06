# -*- coding: utf-8 -*-
"""
@Date : '2020/1/6'
@Desc :  zal
"""
import cv2
import numpy as np
import tensorflow as tf
import os
import random
import time


class CNN(object):
    def __init__(self,
                 image_shape,
                 max_captcha,
                 char_set,
                 models_path,
                 ):
        self.IMAGE_HEIGHT = image_shape[0]
        self.IMAGE_WIDTH = image_shape[1]
        self.MAX_CAPTCHA = max_captcha
        char_set = char_set
        self.CHAR_SET_LEN = len(char_set)
        self.model_path = models_path
        self.reshape_IMAGE_HEIGHT = self.get_out_img_shape(self.IMAGE_HEIGHT)
        self.reshape_IMAGE_WIDTH = self.get_out_img_shape(self.IMAGE_WIDTH)
        self.X = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        self.Y = tf.placeholder(tf.float32, [None, self.MAX_CAPTCHA * self.CHAR_SET_LEN])
        self.keep_prob = tf.placeholder(tf.float32)

    def cnn_net(self, b_alpha=0.1):
        """
        3层卷积神经网络
        """
        x = tf.reshape(self.X, shape=[-1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1])

        wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32],
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64],
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bc2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128],
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)

        wd1 = tf.get_variable(name='wd1', shape=[self.reshape_IMAGE_HEIGHT * self.reshape_IMAGE_WIDTH * 128, 1024],
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, self.reshape_IMAGE_HEIGHT * self.reshape_IMAGE_WIDTH * 128])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, self.keep_prob)

        wout = tf.get_variable('name', shape=[1024, self.MAX_CAPTCHA * self.CHAR_SET_LEN],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(b_alpha * tf.random_normal([self.MAX_CAPTCHA * self.CHAR_SET_LEN]))
        out = tf.add(tf.matmul(dense, wout), bout)
        return out

    def text2vec(self, text):
        text_len = len(text)
        if text_len > self.MAX_CAPTCHA:
            raise ValueError('验证码最长{}个字符'.format(self.MAX_CAPTCHA))
        vector = np.zeros(self.MAX_CAPTCHA * self.CHAR_SET_LEN)

        def char2pos(c):
            if c == '_':
                k = 62
                return k
            k = ord(c) - 48
            if k > 9:
                k = ord(c) - 55
                if k > 35:
                    k = ord(c) - 61
                    if k > 61:
                        raise ValueError('No Map')
            return k

        for i, c in enumerate(text):
            idx = i * self.CHAR_SET_LEN + char2pos(c)
            vector[idx] = 1
        return vector

    def vec2text(self, vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_idx = c % self.CHAR_SET_LEN
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('vec2text  error')
            text.append(chr(char_code))
        return "".join(text)

    def predict(self, image):
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        self.Y = tf.placeholder(tf.float32, [None, self.MAX_CAPTCHA * self.CHAR_SET_LEN])
        self.keep_prob = tf.placeholder(tf.float32)
        output = self.cnn_net()
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        cnn_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with cnn_sess.as_default():
            saver.restore(cnn_sess, tf.train.latest_checkpoint(self.model_path))
        predict_argmax = tf.argmax(tf.reshape(output, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN]), 2)
        if type(image) is str:
            image = cv2.imread(image)
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        image = np.float32(image)
        image = image.flatten() / 255
        text_list = cnn_sess.run(predict_argmax, feed_dict={self.X: [image], self.keep_prob: 1})
        text = text_list[0].tolist()
        vector = np.zeros(self.MAX_CAPTCHA * self.CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * self.CHAR_SET_LEN + n] = 1
            i += 1
        predict_text = self.vec2text(vector)
        return predict_text

    @staticmethod
    def get_out_img_shape(img_shape):
        second = 0
        while True:
            second += 1
            out_img_shape = int(int(img_shape) / 2)
            if int(img_shape) % 2 == 1:
                out_img_shape += 1
            img_shape = out_img_shape
            if second >= 3:
                break
        return out_img_shape

    @staticmethod
    def get_image_file_name(imgFilePath):
        fileName = []
        total = 0
        for filePath in os.listdir(imgFilePath):
            captcha_name = filePath.split('/')[-1]
            fileName.append(captcha_name)
            total += 1
        random.seed(time.time())
        random.shuffle(fileName)
        return fileName, total

    def get_captcha_text_and_image(self, imageFilePath, image_filename_list, imageAmount):
        num = random.randint(0, imageAmount - 1)
        img = cv2.imread(os.path.join(imageFilePath, image_filename_list[num]), 0)
        if img.shape != (self.IMAGE_HEIGHT, self.IMAGE_WIDTH):
            img = cv2.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        img = np.float32(img)
        text = image_filename_list[num].split('_')[0]
        return text, img

    def wrap_get_captcha_text_and_image(self, imageFilePath, image_filename_list, imageAmount):
        while True:
            text, image = self.get_captcha_text_and_image(imageFilePath, image_filename_list, imageAmount)
            if image.shape == (self.IMAGE_HEIGHT, self.IMAGE_WIDTH):
                return text, image

    def get_next_batch(self, imageFilePath, image_filename_list=None, batch_size=32):
        batch_x = np.zeros([batch_size, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        batch_y = np.zeros([batch_size, self.MAX_CAPTCHA * self.CHAR_SET_LEN])

        for listNum in os.walk(imageFilePath):
            pass

        imageAmount = len(listNum[2])
        for i in range(batch_size):
            text, image = self.wrap_get_captcha_text_and_image(imageFilePath, image_filename_list, imageAmount)
            batch_x[i, :] = image.flatten() / 255
            batch_y[i, :] = self.text2vec(text)

        return batch_x, batch_y

    def train(self,
              batch_size,
              train_images_path,
              valid_images_path,
              learning_rate,
              learning_steps
              ):
        # 训练
        output = self.cnn_net()
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        predict = tf.reshape(output, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        image_filename_list, _ = self.get_image_file_name(train_images_path)
        image_filename_list_valid, _ = self.get_image_file_name(valid_images_path)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            while True:
                batch_x, batch_y = self.get_next_batch(train_images_path, image_filename_list, batch_size)
                _, loss_ = sess.run([optimizer, loss],
                                    feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                # 每100 step计算一次准确率
                if step % 100 == 0:
                    batch_x_test, batch_y_test = self.get_next_batch(valid_images_path, image_filename_list_valid,
                                                                     batch_size)
                    acc = sess.run(accuracy,
                                   feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.0})
                    print(step, acc)

                    # 训练结束
                    if step >= learning_steps:
                        saver.save(sess, self.model_path, global_step=step)
                        break
                step += 1
