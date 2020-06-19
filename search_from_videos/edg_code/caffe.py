#encoding ：UTF-8

import numpy as np
import cv2
import os
import traceback
import shutil

class Caffe_detection:
    def __init__(self, max_face_count, min_confidence, model_Path, weight_Path, write_dict="", frequency=1):
        # 初始化内部参数
        if max_face_count < 1:
            self.max_face_count = 1
        else:
            self.max_face_count = max_face_count
        self.faces = []                             # 按照confidence递减存入(face_image,confidence)
        self.write_dict = write_dict
        self.flag = 0                               # 帧采样flag
        self.start_frame_id = -1                    # 目标在视频中起始帧标号（0~n)
        self.frame_length = 0                       # 目标在视频中帧长

        # 初始化caffe识别参数
        self.model_Path = model_Path
        self.weight_Path = weight_Path
        self.min_confidence = float(min_confidence)
        self.frequency = frequency

        self.write_path = os.path.join(self.write_dict, "path.txt")

        # 初始化path.txt
        if os.path.exists(self.write_path):
            os.remove(self.write_path)

        # 初始化模型
        self.caffe_net = cv2.dnn.readNetFromCaffe(self.model_Path, self.weight_Path)

        # 存储上次收到的识别框
        self.rect_buff = [-1, -1, -1, -1]
        self.is_newRect=False

    def write_point_path(self, rect):
        try:
            if os.path.exists(self.write_path):
                fp = open(self.write_path, "a")
                fp.write("," + str(rect[0]) + " " + str(rect[1]) + " " + str(rect[2]) + " " + str(rect[3]))
            else:
                fp = open(self.write_path, "w")
                fp.write(str(rect[0]) + " " + str(rect[1]) + " " + str(rect[2]) + " " + str(rect[3]))
        except Exception as ex:
            print("unexpected caffe.py：%s" % ex)
            print("------------------------------------")
            print(traceback.format_exc())
        finally:
            fp.close()

    # 向磁盘写入置信度较高的人脸图片
    def write(self):
        if not os.path.exists(self.write_dict):
            os.mkdir(self.write_dict)

        # 未检测到人脸时结束检测
        if len(self.faces) < 1 or self.frame_length < 20:
            shutil.rmtree(self.write_dict,True)
            return False

        count = 0
        for face in self.faces:
            img = self.resize_image(face[0])
            if img is not None:
                # cv2.imwrite(os.path.join(self.write_dict, str(count)+".jpg"), img)  # 不支持绝对路径和中文
                cv2.imencode('.jpg', img)[1].tofile(os.path.join(self.write_dict, str(count)+".jpg"))
                count += 1

        if count < 1:
            shutil.rmtree(self.write_dict,True)
            return False

        # 写入目标出现的视频帧范围
        self.write_frame_info()
        return True

    # 检测人脸并放入缓冲区rect=(x0,y0,x1,y1)
    def face_detector(self, row_image, enlar_body_rect=None, row_rect=None):
        row_height, row_width = row_image.shape[:2]

        if row_rect is not None:
            self.rect_buff = row_rect
            self.is_newRect=True

        if self.flag % self.frequency==0:
            self.flag = 0
        else:
            self.flag += 1
            return

        # 输入图片并重置大小符合模型的输入要求
        image = row_image
        if enlar_body_rect is not None:
            image = row_image[enlar_body_rect[1] : enlar_body_rect[3], enlar_body_rect[0] : enlar_body_rect[2]]  # 获取人体部分

        image_height, image_width = image.shape[:2]  # 获取图像的高和宽，用于画图
        if image_width < 100 or image_height < 100:
            return
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

    def write_frame_info(self):
        try:
            fp = open(self.write_path, "a")
            fp.write("\n" + str(self.start_frame_id) + " " + str(self.frame_length) + "\n")
        except Exception as ex:
            print("unexpected caffe.py：%s" % ex)
            print("------------------------------------")
            print(traceback.format_exc())
        finally:
            fp.close()

    def resize_image(self, image, size=80):
        try:
            image_height, image_width = image.shape[:2]
            if image_height <1 or image_width < 1:
                return None

            enlargement_factor = 0.0
            if image_height < image_width:
                enlargement_factor = size / image_height
            else:
                enlargement_factor = size / image_width

            if enlargement_factor > 4.0:
                enlargement_factor = 4.0

            return cv2.resize(image, None, fx=enlargement_factor, fy=enlargement_factor, interpolation=cv2.INTER_LINEAR)
        except Exception as ex:
            print("unexpected caffe.py：%s" % ex)
            print("------------------------------------")
            print(traceback.format_exc())
            return None,False

    def flush(self, frame_index):
        if self.start_frame_id < 0:
            self.start_frame_id = frame_index
        self.frame_length += 1

        if not self.is_newRect:
            self.rect_buff = [-1, -1, -1, -1]

        self.write_point_path(self.rect_buff)
        self.is_newRect=False
