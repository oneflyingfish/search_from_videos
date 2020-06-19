#encoding ：UTF-8

from __future__ import division, print_function, absolute_import

import os
import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from detector import Detector
import traceback
from frames_subtraction import FSB
warnings.filterwarnings('ignore')

### 测试仿真
import matplotlib.pyplot as plt
### 测试仿真

class Processor:
    def __init__(self, yolo, video_path, save_path="result", sample_frequency=50, caffe_confidence=0.5, max_face_sample_count=20, max_object_life=50, show_process=True):
        # 人脸采样参数
        self.sample_frequency = sample_frequency                # 人脸采样n帧每次
        self.caffe_confidence = caffe_confidence                # 人脸最小置信度
        self.max_face_sample_count = max_face_sample_count      # 人脸最大采样数量
        self.max_object_life = max_object_life                  # 对象n帧仍未出现视为消失

        # caffe模型
        self.caffe_model_Path = "model_data/deploy.prototxt.txt"
        self.caffe_weight_Path = "model_data/res10_300x300_ssd_iter_140000.caffemodel"

        # Deep Sort参数
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0

        # 深度关联度量算法[多目标跟踪算法]（Deep Sort）
        self.model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename,batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)

        # 杂项参数
        self.yolo = yolo
        self.video_path = video_path
        self.show_process = show_process
        self.frame_index = -1
        self.fps=0.0

        # 初始化
        # 创建存储文件夹
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.write_path = os.path.join(save_path,os.path.splitext(os.path.basename(self.video_path))[0])
        if not os.path.exists(self.write_path):
            os.mkdir(self.write_path)

        self.video_capture = cv2.VideoCapture(self.video_path)
        self.detect = Detector(self.max_object_life,
                               self.max_face_sample_count,
                               self.caffe_confidence,
                               self.write_path,
                               self.caffe_model_Path,
                               self.caffe_weight_Path,
                               self.sample_frequency)

        # 运行时参数
        self.frame_count = int(self.video_capture.get(7))
        self.show_freq = int(0.01 * self.frame_count)
        # self.process = 0

        ### 测试仿真
        self.times = []
        self.fpss = []
        self.indexs = []
        ### 测试仿真

    def run(self):
        print("start to analyze videos...")

        t1 = time.time()
        try:
            self.fps = 0.0
            while True:
                #读取视频帧
                ret, frame = self.video_capture.read()
                if ret != True:
                    break

                self.frame_index += 1

                image = Image.fromarray(frame[...,::-1])            # bgr to rgb
                boxs = self.yolo.detect_image(image)                # 获取识别框
                # print("box_num",len(boxs))

                features = self.encoder(frame, boxs)
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

                # 非极大值抑制
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # # 显示YOLO检测结果
                # for det in detections:
                #     bbox = det.to_tlbr()
                #     cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

                # 调用追踪器对象
                self.tracker.predict()
                self.tracker.update(detections)

                # Deep Sort跟踪算法
                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()

                    self.detect.insert_detection(track.track_id, frame, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                self.detect.flush(self.frame_index)

                # 计算平均帧率
                if self.show_process:
                    # self.process += 1
                    if self.frame_index % self.show_freq == 0 and (self.frame_index * 100 // self.frame_count) > 0 and self.frame_index < self.frame_count:
                        self.fps = self.frame_index / (time.time() - t1)
                        print("analyze process: %d%%, fps= %.2f.      " % (self.frame_index * 100 // self.frame_count, self.fps), end="\r")

                ### 测试仿真
                self.fps = self.frame_index/ (time.time() - t1)
                self.times.append(time.time() - t1)
                self.fpss.append(self.fps)
                self.indexs.append(self.frame_index / self.frame_count)
                ### 测试仿真

            if self.show_process:
                print("analyze process: 100%%, fps= %.2f.     " % (self.fps))

        except Exception as ex:
            print("unexpected main.py：%s" % ex)
            print("------------------------------------")
            print(traceback.format_exc())
        finally:
            print("finish to analyze videos, it takes %.2fs" % (time.time() - t1))
            self.release()

            ### 测试仿真
            plt.plot(self.times, self.indexs,color='green',label='left')
            plt.xlabel("run time(s)")
            plt.ylabel("executive ratio")
            plt.twinx()
            plt.plot(self.times, self.fpss, color='orange', label='right')
            plt.ylabel("current fps")
            plt.ylim([0.0,6.0])
            plt.show()
            ### 测试仿真

    def release(self):
        if self.detect is not None:
            self.detect.release()
            self.detect = None

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None