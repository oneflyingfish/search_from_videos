#encoding ：UTF-8

import numpy as np
import cv2
import os
import traceback
import shutil

class Caffe_detection:
    def __init__(self, max_face_count, min_confidence, model_Path, weight_Path):
        # 初始化内部参数
        if max_face_count < 1:
            self.max_face_count = 1
        else:
            self.max_face_count = max_face_count
        self.faces = []                             # 按照confidence递减存入(face_image,confidence)

        # 初始化caffe识别参数
        self.model_Path = model_Path
        self.weight_Path = weight_Path
        self.min_confidence = float(min_confidence)

        # 初始化模型
        self.caffe_net = cv2.dnn.readNetFromCaffe(self.model_Path, self.weight_Path)

    # 检测人脸并放入缓冲区rect=(x0,y0,x1,y1)
    def face_detector(self, image):
        image_height, image_width = image.shape[:2]  # 获取图像的高和宽，用于画图
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # 预测结果
        self.caffe_net.setInput(blob)
        detections = self.caffe_net.forward()

        for i in range(0, detections.shape[2]):
            # 获得置信度
            confidence = detections[0, 0, i, 2]

            # 过滤掉低置信度的像素
            if confidence > self.min_confidence :
                # 获得人脸的位置
                box = detections[0, 0, i, 3:7] * np.array([image_width, image_height, image_width, image_height])
                (x0, y0, x1, y1) = box.astype("int")
                # 边框超出图片边界，判定识别错误
                if x0 >= x1 or x1 > image_width or y0 >= y1 or y1 > image_height:
                    continue

                # 人脸存入缓冲区
                self.insert_detection(image[y0:y1, x0:x1], confidence)

    # 向缓冲区加入人脸
    def insert_detection(self,image, confidence=-1):
        if len(self.faces) < 1:
            self.faces.append((image, confidence))
            return

        for i in range(len(self.faces)):
            if self.faces[i][1] < confidence:
                self.faces.insert(i, (image, confidence))
                break

            # 找不到更小置信度的项
            if i >= (len(self.faces) - 1):
                self.faces.append((image, confidence))      # 尾插
                break

        # 删除多余项
        if len(self.faces) > self.max_face_count:
            del (self.faces[-1])
            return
