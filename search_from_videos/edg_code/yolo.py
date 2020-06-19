#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
#from timeit import time
import time
from timeit import default_timer as timer                       # 用于计算帧速率

import numpy as np
import keras
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5'                  # yolov3预训练权重
        self.anchors_path = 'model_data/yolo_anchors.txt'       # 集合
        self.classes_path = 'model_data/coco_classes.txt'       # 识别类型
        self.score = 0.5
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)                    # fixed size or (None, None)
        # self.model_image_size = (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)    # 将相对路径转化为绝对路径（完整路径格式）
        # 获取可识别类型列表
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]          # 去除列表元素首尾空格
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)          # 转化为两列的矩阵
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        # 断言语句，非keras模型时直接结束程序
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file, try to download yolov3.weight from https://pjreddie.com/media/files/yolov3.weights, then run "python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5"'

        # keras加载模型
        self.yolo_model = keras.models.load_model(model_path, compile=False)
        print('model:{0}, anchors:{1}, classes:{2} loaded.'.format(self.model_path,self.anchors_path,self.classes_path))

        # 为每种类型物体生成不同的边框颜色
        hsv_tuples = [(x / len(self.class_names), 1.0, 1.0) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 打乱边框颜色
        random.seed(10101)              # 固定随机种子，使得每次运行都获得一致的边框颜色
        random.shuffle(self.colors)     # 打乱边框颜色，消除其与识别物体类型顺序之间的关系
        random.seed(None)               # 恢复随机种子

        # 构建模型
        self.input_image_shape = K.placeholder(shape=(2, ))                     # 输入图片为（2,n)的矩阵
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,len(self.class_names), self.input_image_shape,score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # 获取检测到的物体边框集合     
    def detect_image(self, image):
        ''' 探测图像 '''
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 计算预测值，对 物体->对象 的映射可能性打分
        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],feed_dict={self.yolo_model.input: image_data,self.input_image_shape: [image.size[1], image.size[0]],K.learning_phase(): 0})
        
        return_boxs = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]   # 获取检测到的物体类型
            if predicted_class != 'person' :
                continue

            # 存储物体边框信息
            box = out_boxes[i]
            # score = out_scores[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])

            # 物体到达画面边界时特殊处理
            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0
            return_boxs.append([x,y,w,h])

        return return_boxs

    def close_session(self):
        self.sess.close()
