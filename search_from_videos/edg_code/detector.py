#encoding ：UTF-8

# 管理器，监测人物对象群体

from caffe import Caffe_detection

import os
import cv2
import numpy as np

class Detector:
    def __init__(self,life=50,max_face_count=5, min_confidence=0.5, write_dict="detect_faces", model_Path = "model_data/deploy.prototxt.txt", weight_Path = "model_data/res10_300x300_ssd_iter_140000.caffemodel",frequence=50):
        # 初始化内部参数
        if max_face_count < 1:
            self.max_face_count = 1
        else:
            self.max_face_count = max_face_count
        self.write_dict = write_dict
        self.life = life

        # 初始化caffe识别参数
        self.model_Path = model_Path
        self.weight_Path = weight_Path
        self.min_confidence = float(min_confidence)
        self.frequence = frequence

        self.count = 0                              # 记录程序启动后总共记录的对象个数
        self.buffer = set()
        self.detections = {}

    # 获取内存缓冲区中记录的对象个数
    def get_current_count(self):
        return len(self.detections.keys)

    def insert_detection(self, id, row_image, rect=None):
        self.buffer.add(id)

        enlargement_factor = self.resize(row_image, rect, 300, 600)
        # enlargement_factor = 3

        image = cv2.resize(row_image, None, fx=enlargement_factor, fy=enlargement_factor, interpolation=cv2.INTER_LINEAR)
        enlar_rect = [int(enlargement_factor * i) for i in rect]

        if id not in self.detections.keys():
            self.count += 1
            path = os.path.join(self.write_dict, str(self.count))
            if not os.path.exists(path):
                os.mkdir(path)

            obj = Caffe_detection(self.max_face_count, self.min_confidence, self.model_Path, self.weight_Path, path, self.frequence)
            obj.face_detector(image, enlar_rect, rect)
            self.detections[id] = [obj, 0]
            return
        else:
            self.detections[id][0].face_detector(image, enlar_rect, rect)

    # 哨兵函数，记录并删除长期未出现的对象
    def flush(self, frame_index):
        remove_list=[]
        for i in self.detections.keys():
            self.detections[i][0].flush(frame_index)        # 更新对象中帧号
            if i in self.buffer:
                self.detections[i][1] = 0                   # 重置生命周期
                continue
            else:
                if self.detections[i][1] >= self.life:
                    self.detections[i][0].write()
                    remove_list.append(i)
                    continue
                else:
                    self.detections[i][1] += 1              # 年龄自增
                    continue
        for i in remove_list:
            self.detections.pop(i)
        self.buffer.clear()

    def release(self):
        for i in self.detections.keys():
            self.detections[i][0].write()
        self.detections.clear()
        self.buffer.clear()

    def resize(self, image, rect, min_width=400, min_height=800):
        if rect is None:
            return 1.0

        enlargement_factor1 = min_width / (rect[2] - rect[0])
        enlargement_factor2 = min_height / (rect[3] - rect[1])

        enlargement_factor = 0.0
        if enlargement_factor1 > enlargement_factor2:
            enlargement_factor = enlargement_factor1
        else:
            enlargement_factor = enlargement_factor2

        if enlargement_factor < 5:
            return enlargement_factor
        else:
            return 5