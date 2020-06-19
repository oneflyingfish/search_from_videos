#encoding ：UTF-8

import cv2
import os
import numpy as np
import time
import traceback
import gc

class Face_Model_Creator:
    def __init__(self, dict):
        self.dict = dict

    def run(self):
        print("start to craate face model, it may take a little time...")
        t1 = time.time()

        try:
            video_dicts = os.listdir(self.dict)
            for video_dict_name in video_dicts:
                video_dict = os.path.join(self.dict, video_dict_name)
                if not os.path.isdir(video_dict):
                    continue

                object_dicts = os.listdir(video_dict)
                for object_dict_name in object_dicts:
                    object_dict = os.path.join(video_dict, object_dict_name)
                    if not os.path.isdir(object_dict):
                        continue

                    faces, labels = self.get_train_data(object_dict)

                    # 创建EigenFace模型文件
                    Eigenface_model = cv2.face.EigenFaceRecognizer_create()
                    Eigenface_model.train(faces, labels)
                    Eigenface_model.write(os.path.join(object_dict, "eigen_face_model.xml"))

                    # 创建LBPHFace模型文件
                    LBPHface_model = cv2.face.LBPHFaceRecognizer_create()
                    LBPHface_model.train(faces, labels)
                    LBPHface_model.write(os.path.join(object_dict, "lbph_face_model.xml"))

        except Exception as ex:
            print("unexpected caffe.py：%s" % ex)
            print("------------------------------------")
            print(traceback.format_exc())
        finally:
            print("creating face models is ok, take %.2fs"%(time.time()-t1))

    def get_train_data(self, path_dict):
        faces = []
        labels = []

        files = os.listdir(path_dict)
        for file in files:
            img_path = os.path.join(path_dict, file)
            if os.path.isdir(img_path) or not file.endswith(".jpg"):
                continue

            # face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)     # 绝对路径和中文支持问题
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # face=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            face = cv2.resize(face, (200, 200))
            faces.append(np.asarray(face, dtype=np.uint8))
            labels.append(0)

        np.asarray(labels, dtype=np.int32)

        return np.asarray(faces),np.asarray(labels)
