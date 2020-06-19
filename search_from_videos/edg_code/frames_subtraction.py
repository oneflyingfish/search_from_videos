#encoding ：UTF-8

import os
import cv2
import numpy as np
import traceback
import time

# ### 测试仿真
# import matplotlib.pyplot as plt
# ### 测试仿真

class FSB:
    def __init__(self, video_path=None, save_fps=-1, max_leisure=100, confidence=0.0008, save_dict="pre_videos",show_process=True):
        # 处理参数
        if video_path is not None:
            self.video_path = video_path
            self.vc = cv2.VideoCapture(video_path)
        else:
            self.video_path = "camera"
            self.vc = cv2.VideoCapture(0)

        self.max_leisure = max_leisure
        self.show_process = show_process

        # 写入参数
        self.save_dict = save_dict
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        # 运行时参数
        self.frame_width = int(self.vc.get(3))
        self.frame_height = int(self.vc.get(4))
        self.frame_count = int(self.vc.get(7))
        if save_fps < 0:
            self.save_fps=int(self.vc.get(5))
        else:
            self.save_fps = int(save_fps)
        self.show_freq = int(0.01 * self.frame_count)
        self.process = 0

        self.confidence = confidence
        self.frame1 = None
        self.frame2 = None
        self.frameDelta1 = None  # 差分图像1
        self.frameDelta2 = None  # 差分图像2

        self.count = 0
        self.flag = max_leisure
        self.on_writting = False
        self.length = 0

        self.out = None
        self.current_path = ""
        self.min_length = 100

        # 创建存储文件夹
        if not os.path.exists(self.save_dict):
            os.mkdir(self.save_dict)

        # ### 测试仿真
        # self.write_count = 0
        # self.times = []
        # self.compression_radio = []
        # self.processes = []
        # ### 测试仿真

    def run(self, show_process=True):
        print("start to compress the videos, the frame count is %d."%self.frame_count)
        t1 = time.time()

        try:
            while self.vc.isOpened():
                ret, frame3 = self.vc.read()

                if not ret:
                    self.flag = 0
                    if self.out is not None:
                        self.out.release()
                        self.out = None
                        self.on_writting = False
                        self.deal_video()
                    break

                if self.show_process:
                    self.process += 1
                    if self.process % self.show_freq == 0 and (self.process * 100 // self.frame_count) > 0 and self.process < self.frame_count:
                        print("compress process: %d%%.        " % (self.process * 100 // self.frame_count), end="\r")

                # # 显示视频
                # cv2.imshow("frame", frame3)
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break

                if self.frame2 is None:
                    self.frame2 = frame3
                    continue
                else:
                    self.frameDelta2 = cv2.absdiff(frame3, self.frame2)  # 帧差2
                    if self.frame1 is None:
                        self.frame1 = self.frame2
                        self.frame2 = frame3
                        self.frameDelta1 = self.frameDelta2
                        continue

                thresh = cv2.bitwise_and(self.frameDelta1, self.frameDelta2)  # 图像与运算
                thresh2 = thresh.copy()

                # 为下次识别初始化准备
                self.frame1 = self.frame2
                self.frame2 = frame3.copy()
                self.frameDelta1 = self.frameDelta2

                # 结果转为灰度图
                thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
                # 图像二值化
                thresh = cv2.threshold(thresh, 8, 255, cv2.THRESH_BINARY)[1]

                '''
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 椭圆结构
                # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))   # 十字结构
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))    # 矩形结构
                '''

                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))     # 十字结构
                thresh = cv2.dilate(thresh, kernel, iterations=1)               # 十字膨胀
                thresh = cv2.erode(thresh, kernel, iterations=1)                # 十字腐蚀
                thresh = cv2.dilate(thresh, kernel, iterations=1)               # 十字膨胀
                thresh = cv2.erode(thresh, kernel, iterations=1)                # 十字腐蚀

                # 阀值图像上的轮廓位置
                cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                max_area = 0
                # 遍历轮廓
                for c in cnts:
                    if cv2.contourArea(c) > max_area:
                        max_area = cv2.contourArea(c)

                # 忽略小轮廓，排除误差
                if max_area < self.confidence * frame3.shape[0] * frame3.shape[1]:
                    if self.on_writting:
                        self.flag += 1
                        self.out.write(frame3)
                        self.length += 1

                        # ### 测试仿真
                        # self.write_count += 1
                        # ### 测试仿真
                    else:
                        self.flag = 0

                    if self.flag > self.max_leisure:
                        if self.on_writting:
                            self.out.release()
                            self.on_writting = False
                            self.flag = 0
                            self.out = None
                            self.deal_video()
                else:
                    if not self.on_writting:
                        self.current_path = os.path.join(self.save_dict,os.path.splitext(os.path.basename(self.video_path))[0]+str(self.count)+".avi")
                        self.out = cv2.VideoWriter(self.current_path, self.fourcc,self.save_fps, (self.frame_width, self.frame_height))
                        self.count += 1
                        self.on_writting = True
                        self.length = 0

                    self.flag = 0
                    self.out.write(frame3)
                    self.length += 1
                    # print(self.length)

                    # ### 测试仿真
                    # self.write_count += 1
                    # ### 测试仿真

                # ### 测试仿真
                # self.compression_radio.append(self.write_count / self.process)
                # self.processes.append(self.process / self.frame_count)
                # self.times.append(time.time() - t1)
                # ### 测试仿真

            if self.show_process:
                print("compress process: 100%.           ")



        except Exception as ex:
            print("unexpected frames_subtraction.py：%s" % ex)
            print("------------------------------------")
            print(traceback.format_exc())
        finally:
            if self.out is not None:
                self.out.release()
                if self.length < self.min_length:
                    os.remove(self.current_path)
            self.vc.release()
            print("finish to compress the videos, it takes %.2fs" % (time.time() - t1))
            # cv2.destroyAllWindows()

            # ### 测试仿真
            # plt.plot(self.times[3:], self.compression_radio[3:],color='green',label='left')
            # plt.xlabel("run time(s)")
            # plt.ylabel("compression_radio")
            # plt.twinx()
            # plt.plot(self.times[3:], self.processes[3:], color='orange', label='right')
            # plt.ylabel("executive ratio")
            # plt.show()
            # ### 测试仿真

    def deal_video(self):
        if not os.path.exists(self.current_path):
            return

        if self.length < self.min_length:
            os.remove(self.current_path)