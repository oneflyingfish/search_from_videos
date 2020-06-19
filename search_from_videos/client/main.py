#encoding ：UTF-8

import cv2
import os
import numpy as np
import time
import traceback
import shutil
import numpy as np
from caffe import Caffe_detection

# 参数设置
image_path = "input.jpg"        # 待搜索任务图片，应包含清晰人脸
face_model_kind = 1             # 0：Eigen face  1: LBPH face
t = 0.9                         # 阈值

# 路径设置, 取决于服务器运算结果，此处如果报错路径可改为绝对路径
pre_dict = "search_from_videos/pre_videos"
result_dict = "search_from_videos/result"
search_result_path = "search_from_videos/search_result"

# caffe模型参数
model_Path = "model/deploy.prototxt.txt"
weight_Path = "model/res10_300x300_ssd_iter_140000.caffemodel"

def output_video(video_path, config_path, save_path="search_result", save_fps=25):
    vc = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # 读入配置文件
    fp = open(config_path, "r")
    path_list = fp.readline().split(",")
    frame_info = fp.readline().split(" ")
    fp.close()

    start_frame, frame_length = int(frame_info[0]), int(frame_info[1])
    frame_width = int(vc.get(3))
    frame_height = int(vc.get(4))

    out = cv2.VideoWriter(save_path, fourcc, save_fps, (frame_width, frame_height))

    # 跳过前帧
    if start_frame>0:
        for i in range(start_frame):
            ret, frame = vc.read()
            if ret != True:
                out.release()
                vc.release()
                return False

    if frame_length < 1:
        out.release()
        vc.release()
        return False

    for i in range(frame_length):
        #读取视频帧
        ret, frame = vc.read()
        if ret != True:
            out.release()
            vc.release()
            return False

        rect = path_list[i].split(" ")
        if int(rect[2]) > 0 and int(rect[3]) > 0:
            cv2.rectangle(frame, (int(rect[0]), int(rect[1])),(int(rect[2]), int(rect[3])), (65, 241, 149), 1)
        out.write(frame)

    out.release()
    vc.release()


def main():
    predict_list = []

    # 运行时参数
    result_count = 0

    # 初始化
    if not os.path.exists(search_result_path):
        os.mkdir(search_result_path)

    # caffe模型
    caffe_model_Path = "model/deploy.prototxt.txt"
    caffe_weight_Path = "model/res10_300x300_ssd_iter_140000.caffemodel"

    # 从图片中提取人脸
    try:
        # image = cv2.imread(image_path)      # 不支持绝对路径和中文
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if image is None:
            print("error: no input.")
            return

        image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)

        caf = Caffe_detection(max_face_count=1, min_confidence=0.3, model_Path=caffe_model_Path, weight_Path=caffe_weight_Path)
        caf.face_detector(image)
        if len(caf.faces) < 1:
            print("please input a picture contain only one face")
            return

        gray_face = cv2.cvtColor(caf.faces[0][0], cv2.COLOR_BGR2GRAY)
        image = cv2.resize(gray_face, (200, 200), interpolation=cv2.INTER_LINEAR)

        video_dicts = os.listdir(result_dict)
        for video_dict_name in video_dicts:
            video_dict = os.path.join(result_dict, video_dict_name)
            if not os.path.isdir(video_dict):
                continue

            object_dicts = os.listdir(video_dict)
            for object_dict_name in object_dicts:
                object_dict = os.path.join(video_dict, object_dict_name)
                if not os.path.isdir(object_dict):
                    continue

                # 读取Face模型文件
                if face_model_kind == 0:
                    face_model = cv2.face.EigenFaceRecognizer_create()
                    face_model.read(
                        os.path.join(object_dict, "eigen_face_model.xml"))
                else:
                    face_model = cv2.face.LBPHFaceRecognizer_create()
                    face_model.read(os.path.join(object_dict, "lbph_face_model.xml"))

                predict = face_model.predict(image)
                predict_list.append((predict[1], os.path.join(pre_dict, video_dict_name + ".avi"), os.path.join(object_dict, "path.txt")))

                # if predict[0] < 0:
                #     continue

                # # 输出视频
                # file_name = os.path.join(search_result_path, str(result_count) + ".avi")
                # output_video(os.path.join(pre_dict, video_dict_name + ".avi"), os.path.join(object_dict, "path.txt"), save_path=file_name)
                # result_count += 1

                # fp = open(os.path.join(search_result_path, "info.txt"), "a")
                # fp.write("%.3f %s\n" % (predict[1], file_name))
                # fp.close()

        # 存储所有存储结果
        predict_list.sort(key=lambda x: x[0])  # 根据置信度距离，升序排序,reverse=True将降序

        fp = open(os.path.join(search_result_path, "info.txt"), "w")
        for i in predict_list:
            fp.write("%.3f %s %s\n" % (i[0],i[1],i[2]))
        fp.close()

        # 置信度小于平均水平时直接生成视频
        confidence_mean = np.mean([x[0] for x in predict_list])      # 求解平均值
        for pre in predict_list:
            if pre[0] < confidence_mean*t:
                file_name = os.path.join(search_result_path, str(result_count) + ".avi")
                output_video(pre[1], pre[2], save_path=file_name)
                result_count += 1

    except Exception as ex:
        print("unexpected caffe.py：%s" % ex)
        print("------------------------------------")
        print(traceback.format_exc())

if __name__ == '__main__':
    main()
    print('search finished, see the result in "%s."\n' % search_result_path)